# 07. その他のホットスポット — sampler / verify / PLE / GR / ランタイム

> 対象: `src/kernels/cuda/sampler.cu`（1393）、`verify_kernels.cu`（1357）、`ple.cu`（390）、
> `native_ple_postops.cu`（289）、`gr.cu`（500）、`fused_gr.cu`（1991）、`elementwise.cu`（446）、
> `cvec.cu`（174）、`src/core/device.cu`（364）、`pinned.cu`（800）、`remote_expert_opt.cu`（197）

各章で拾えなかった「地味だが事故が起きる場所」をまとめる。ここも**コメントが最も饒舌な領域**で、
バグの歴史がそのまま設計理由になっている。

---

## 1. `sampler.cu` — 「順序がこのファイルの全内容」

冒頭 1 行目からして:

> 「`sampler.cu` — P2.S2: サンプラ連鎖、**llama.cpp の順序で**。
> `penalties -> top_k -> top_p -> min_p -> temperature -> pick`」

### 1.1 何が難しかったのか

> 「**順序がこのファイルの全内容である。** llama.cpp は `params.samplers` を歩いて連鎖を作る。
> 既定は `{PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE}`
> （3cf03257 の `common/common.h`）— **penalties 段は 1 つだけ、最初に**、そして
> **TEMPERATURE は切り詰めフィルタの後**。
> （Issue #53: このファイルは以前、temperature の後で penalties を 2 度掛け、top_p の前に min_p を掛けていた —
> どちらも `common/sampling.cpp` の `case` ラベルの順序から取ったもので、**連鎖が走る順序ではない**。）」

決定的な一文が続く:

> 「**どんな順序でも妥当なトークンを出す。だから分布レベルの比較しか違いを見分けられない** —
> parity テストはそれを独立に計算した分布と突き合わせて行う。」

これが「静かに間違う」の最良の例だ。1 トークンを見ている限り、penalties を 2 度掛けても
「それらしい文章」が出る。**分布**を見ないと分からない。

### 1.2 3 つの実装が同じトークンを選ぶ

> 「greedy は素の argmax（1 ブロック/トークン、語彙全体。sm_90+ でペナルティ無しなら
> `sampler_greedy_cluster_kernel`: 行あたり 8 CTA のクラスタで同じトークン）。サンプル連鎖には
> **同じトークンをビット単位で選ぶ 3 つの実装**がある:
> — 分割 top_k（既定）: `sampler_split_part_kernel` が各行を 4,096 ロジットのブロックに GPU 全体で割り、
> 各ブロックが自分の top_k を保ち、`sampler_split_merge_kernel` がそれらを併合して残りを走らせる
> — `sampler_one_block_kernel`（`STRATA_SAMPLER_ONE_BLOCK=1`、分割が走れないときのフォールバック）:
> 1 ブロック/トークン、top_k 回のブロック argmax
> — `sampler_kernel`（`STRATA_OLD_SAMPLER=1`）: engine 0.1.20 のカーネル、参照として保持
> 新しい 2 つは `sampled_tail_warp`（top_p / min_p / temperature / 抽選を 1 ワープで）を共有する。」

### 1.3 Philox を選んだ理由（再現性の設計）

> 「Philox 4x32-10、フェーズが要求するカウンタベース生成器。
> **カウンタベースであることが重要**: 乱数列が「何回引いたか」ではなく **(seed, position) の関数**になるので、
> バッチをどんな順でサンプルしてもよく、実行が再現可能になる。」

「GPU で並列にサンプルする」ときに最初に壊れるのは再現性だ。カウンタベースなら
「トークン位置 → 乱数」が関数になる。**これは [05](05_attention_gdn.md) のキャプチャ監査と同じ思想**で、
「状態を隠さない」。

### 1.4 その他の段

`coupled_stage` / `coupled_penalize` は投機デコードの「結合」サンプリング
（`coupled_draft.hpp` / `spec_prob.hpp`）。投機では draft と target の分布の一致が要るので、
乱数も分布も共有する必要がある（[07 §2](#2-verify_kernelscu--投機の検証) と対）。


---

## 2. `verify_kernels.cu` — 投機の検証

### 2.1 前提となる設計

Strata は MTP 層で最大 3 トークンを先読みし、**1 パスで全部検証**する（2.4–3.2 トークン/パス）。
繰り返し入力では prompt lookup が最大 5 トークンを先読みする。

検証の意味は「draft が提案した k トークンを、**素の decode と同じ計算**で正しいか確かめる」こと。
ここで 1 ビットでも違うと**受理率が静かに落ちる**（プロンプトは同じ、速度だけ落ちる）。

### 2.2 だから「書き写し」である

> 「ここの全カーネルの**トークン毎の演算は、単一トークン版の元（`fused_gdn.cu`、`elementwise.cu`）から
> 書き写されており、同じ演算順序を持つ。だから検証ウィンドウは plain decode をビット単位で再現する。**」

工夫ではなく**制約**を守るコード。具体的には:

| 仕組み | 内容 |
|---|---|
| `gdn_conv_l2_multi_kernel` | 履歴を「読んでから自分のチャネルを書く」— `commit` が 1 なら [hist1, hist2, x_0] を残す |
| state-only commit | 状態の 128 列を **1 ヘッドあたり 4 ブロック×32** に分割（48×4 = 192 ブロック × 128 スレッド、対 48 ブロック × 512）。sq / o / norm / z / y を計算しない |
| なぜ分割か | 旧実装は層の状態 2 MB を約 60 GB/s で読んでいた（帯域の無駄） |
| S25 `STRATA_GDN_SPLIT=1` | 上の分割を有効化 |
| S26 `OUT=false` | 出力縮約とバリア 2 つを落とす。コード形状は保つのでビット一致。ただし **128 スレッド版は約 7% の出力で 1 ulp 違った**（＝並列度を変えると最後のビットが変わる） |
| S26 prefetch | gate/beta/v を先に、次の行の k/q を現在行の間に。gfx1151・h_v 48 で 44.5→39.1 µs（1 行）、47.4→40.5（2 行）、53.8→42.9（4 行）。ただし出力ありではレジスタ圧で負ける |
| `STRATA_QFUSE=1` | q8_1 イメージを `native_quantize_q8_1_kernel` で作る |
| `STRATA_DBG_GDN=1` | #937: **起動が決して許さなかったウィンドウ数を報告し、commit を飛ばす**（そのまま commit するとメモリアパーチャ違反になる） |

### 2.3 「ウィンドウ数」という概念

投機では「何トークン分の状態を検証するか」をウィンドウと呼ぶ。ウィンドウは
**起動時に決まる**ので、カーネルは「自分が担当するウィンドウ」を受け取る。
#937 のデバッグフラグは「起動が許さなかった数」を報告する — つまり
**「呼び出し側とカーネル側でウィンドウ数の解釈がずれた」**というバグの痕跡で、
それが起きたときは commit を飛ばして**壊れた状態を書かない**ようにしてある。

---

## 3. `ple.cu` — 28.8 GB の n-gram テーブルを扱う層

PLE は **SSD 上の 28.8 GB の n-gram テーブル**（[01](01_memory.md)）。トークンごとに数行だけを
**非バッファ読み**する。`--ple-io direct` が既定。

`ple.cu` は「PLE ブロックの GPU 半分」（P2.S4）。

### 3.1 レガシー演算は「キャプチャした CPU ggml グラフ」に従う

> 「下のレガシー演算は、キャプチャした CPU ggml グラフ（`bench/micro/ple_in.bin` / `ple_out.bin`、
> `ple_layer_xcheck.cpp` が生成）に従う:
> — `rms_norm` は `(double)(x*x)` を**積を先に f32 で丸めてから**累積する — これは文字通り
>   `ggml_compute_forward_rms_norm_f32` がやっていること（`sum += (ggml_float)(x[i00]*x[i00])`、
>   `ggml_float` = double）であり、f32 で累積するのとも、積む前に拡張するのとも違う
> — `silu` は f32 の `x / (1 + expf(-x))`、つまり `ggml_silu_f32`
> — gate の `sqrt` と `sigmoid` は f32」

**「CPU 実装の丸め順序を再現する」**ことが要件。これは [02](02_quant.md) の
「同じ整数を、より少ない命令で」と同じ種類の要求で、**ビット一致のためには順序が仕様**。

### 3.2 3 つの経路

| 経路 | 内容 |
|---|---|
| legacy | 上の CPU グラフ準拠（既定） |
| native BF16（opt-in） | `ple_value` **だけ**をピン留め CUDA の単一トークン BF16/F32 MMVF に置換 |
| native postops（別 opt-in） | 両方の射影の後、ピン留め CUDA の演算に従う |

**置換は 1 関数ずつ**。だから「どこが変わったか」が常に言える。

### 3.3 転置の罠（実際に一度間違えた）

> 「`ref/ngram.py` の (4, 10240) numpy の `kernel[k][c]` は、そのビューの**転置**である。
> ソース自身のビュー（`ggml_view_2d(model.layers[il].ple_conv1d, 1, hc_dim, nb[1], k*nb[0])`）が
> どちらを意味するかを決める。」

さらに `ple_layer_xcheck.cpp` L79–83 には「**これを一度間違えた記録がある**」と書いてある。
ヘッダ `ple.hpp` の重要な発見は `normalized`-**が** conv の入力だということ（名前から逆を想像しやすい）。

PLE のカーネル群: `history_advance`, `gnorm`, `gate`, `bcast`, `conv`, `add3`,
`bf16_gemv`, `to_bf16`（`ple.cu`）、`native_ple_postops.cu` 側に
`gate` / `broadcast` / `conv_residual` / `rms_rep` / `rms_rep_rt` と `*_batch` 版。
定数は `N = 2560`, `H = 4`, `D = 10240`, `HISTORY = 9`。


---

## 4. GR（ゲート付き残差）— `gr.cu` と `fused_gr.cu`

Qwen3.8-Flash-Next のハイパーコネクション（HC = 4 ストリーム）を畳む層。
`fused_gr.cu`（1991 行）は `launch_bounds` カーネルの寄せ集めで、寸法は固定:

```
N = 2560   （n_embd）
HC = 4     （ストリーム数）
D = N * HC = 10240
LR = 320   （hc_lr）
DOWN_BLOCKS = LR / WARPS = 40   （8 行ずつ、注入行のため +1）
UP_COLS = 32、UP_BLOCKS = N / UP_COLS = 80   （up ブロックの d 列 × 4 ストリーム = 128 行）
```


```c
// 遅延隠蔽の norm/up（gr_norm_fast_kernel / gr_up_fast_kernel, STRATA_GR_FAST）:
// gfx906 のものだが、CUDA 向けにもビルドされる
#if defined(STRATA_HIP_GFX906) || !defined(__HIPCC__)
#define STRATA_GR_FAST_BUILD 1
#else
#define STRATA_GR_FAST_BUILD 0
#endif
```

このマクロは [06 §4](06_prefill.md) の教訓の別バージョン: **AMD の特定アーキで書いた最適化を
CUDA 側にもビルドする**とき、両方のプリプロセッサで同じ定義になるようにしている。

`gr.cu`（500 行）は分割版（`gr_norm_kernel`, `gr_down_kernel`）、
`native_gr_norm.cu` は `weighted_rms_norm`、`native_gr_postops.cu` は
`down_silu` / `pre_gated` / `post` / `post_multi`。
**fused と unfused が両方あるのは parity の参照のため**（[04](04_moe.md) の GDN と同じ理由）。


---

## 5. `elementwise.cu` と doorbell

地味なカーネルの集まり（446 行）だが、**GPU と CPU を重ねる仕組みの心臓**がある。

| カーネル | 役割 |
|---|---|
| `embedding_gather_kernel` | 量子化埋め込みの行取り出し（code_bits 2/4/8、`__fmul_rn` で明確に丸め） |
| `gdn_gate_kernel` | `softplus(alpha + dt) * ssm_a` |
| `scale`, `add`, `to_f16`, `to_bf16`, `silu`, `rms_norm_weighted` | 基本演算 |
| `doorbell_ring`, `doorbell_wait`, `doorbell_publish`, `doorbell_publish_res`, `doorbell_publish_value` | **GPU → CPU の鐘** |
| `copy_from_mapped`, `copy_rows_from_mapped`, `copy_from_mapped_multi`, `copy_i32_from_mapped`, `scatter_rows` | マップ済みホストメモリとのコピー |

### doorbell が必要な理由（[00](00_overview.md) の再掲・詳細）

split graph で「attention 側が終わったら CPU プールを走らせてほしい」と GPU が CPU に伝える必要がある。
本来は `cudaEventRecord` でよい。**しかし CUDA グラフのキャプチャ中、`cudaEventRecord` は黙って捨てられる**
（round 199）。だから:

> 「doorbell = マップ済みピンドメモリに**シーケンス番号を書くカーネル**」

CPU はその番号をポーリングする。イベントではなく**データ**なので、グラフに焼いても消えない。
これが [00](00_overview.md) の split graph の「鐘」の実体である。


### softplus の分岐が「化粧ではない」理由

```c
/// ggml_compute_softplus_f32: log1p(exp(x))、大きな x でのオーバーフローを避ける分岐つき。
/// 分岐は化粧ではない。exp(89) は f32 でオーバーフローし、exp(20) は既に 4.85e8 で
/// log1p は相対精度を失う。20 を超えると f32 の範囲では関数は x である。
__device__ __forceinline__ float softplus_dev(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
```

---

## 6. `cvec.cu` — Hadamard 回転（KV の int8 量子化の前処理）

`fwht256`（fast Walsh–Hadamard transform、256 要素）と `cvec_apply`。
KV を q8/q4 に圧縮する前に回転を掛けると、外れ値が分散して量子化誤差が減る
（[05](05_attention_gdn.md) の KV 階層）。**回転は直交なので内積が変わらない** — これが
「量子化しても attention の結果がほぼ変わらない」ことの根拠である。


---

## 7. ランタイム — `device.cu` / `pinned.cu` / `remote_expert_opt.cu`

### 7.1 未初期化メモリを 0 ではなく NaN で埋める

```c
// ゼロではなく NaN パターン。未初期化メモリから読んだ 0 は、量子化解除済み重みや
// マスク済み attention スコアの本物の 0 と区別がつかない — まさに Phase 1 の
// ハーネスが捕まえ続けた「間違っているが尤もらしい値」の類である。
__global__ void poison_kernel(float* p, uint64_t n_floats) {
    const uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_floats) p[i] = __int_as_float(0x7fc00000);
}
```

**0 は「正しい値」と区別できない。NaN は伝播する。** デバッグビルドの基本だが、
この一文がその理由を書いている。

### 7.2 アーキテクチャの取り違えを起動時に検出

`device.cu` は `STRATA_HIP_ARCHS`（`cmake/hip_backend.cmake` が設定）を要求し、
未定義ならビルドを止める。`compiled_for(arch)` は「このバイナリは実行中の GPU 向けにビルドされたか」を
判定する（`"gfx1201:sramecc-:xnack-"` → `"gfx1201"` に切り詰めてから比較）。
[06 §4](06_prefill.md) の WMMA の話と同じ根: **コンパイラマクロでは判定できないので、
CMake が確定したリストを埋め込む**。


### 7.3 `pinned.cu`（800 行）— 42.9 GB を並列に読む

> 「Loader fix: `fseek`/`ftell` は Windows では既定で 32 ビット（そしてパックは 42.9 GB）、
> 64 ビットの綴りはエンジンがビルドする 2 つのプラットフォームで同じではない。」

```c
#ifdef _WIN32
#define STRATA_FSEEK64(f, o) _fseeki64((f), (long long) (o), SEEK_SET)
#else
#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)
#endif
```

さらに `MAP_HUGE_2MB`（2 MB ヒュージページ）を自前で定義（カーネルヘッダに無い場合がある）、
`std::thread` で expert を並列ロード、`std::atomic` で進捗。**6 機種・3 OS で動くコードの現実**。

### 7.4 `remote_expert_opt.cu`

ピア GPU / リモート GPU の expert を扱う経路（[01](01_memory.md)）。
「VRAM キャッシュ → RAM → ピア GPU」の 3 番目の選択肢で、NVLink / P2P がある場合だけ得になる。

---

## 8. この章のまとめ

| 主題 | 一番の教訓 |
|---|---|
| sampler | **どんな順序でも妥当なトークンが出る** → 分布でしか検証できない |
| verify | **書き写す**ことが仕様。並列度を変えると最下位ビットが変わる |
| PLE | CPU 実装の丸め順序が仕様。転置は一度間違えている |
| GR | AMD 向け最適化を CUDA にもビルドする際のマクロ設計 |
| elementwise | **グラフキャプチャでイベントは消える、データは消えない** |
| ランタイム | 未初期化は 0 でなく NaN で埋める；サイズは型で壊れる |

### 読者チェックリスト

1. サンプラの順序が違っても「動く」のはなぜか。どうやって検出するか
2. 投機の検証カーネルが「最適化」ではなく「書き写し」である理由
3. PLE の rms_norm が**積を f32 で丸めてから double で累積**するのはなぜか
4. `cudaEventRecord` がグラフキャプチャで消えるとなぜ doorbell が必要になるか
5. 未初期化メモリを 0 で埋めると何が見えなくなるか
6. Windows の `fseek` が 42.9 GB パックで壊れる理由

