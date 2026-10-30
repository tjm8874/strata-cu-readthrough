<div align="center">

# Strata CUDA ソース読解

### 25,371 行の .cu が、125B MoE を 12GB の VRAM で動かすまで

Strata 0.1.49 / Qwen3.8-Flash-Next (125B, 48層, 24,576 expert, top-10)
</div>

---

## 0. この本の読み方

このリポジトリ src/ 配下の .cu は **46 ファイル・25,371 行**（src/kernels/cuda）、
さらに prefill 用に約 5,000 行（src/prefill）がある。全部を頭から読むのは不可能なので、
この本は「**1 トークンが生まれるまで**」という 1 本の線に沿って並べ直してある。

| 章 | 内容 | 対応するソース |
|---|---|---|
| [00 全体像](00_overview.md) | 実行モード、レイヤ構成、ホスト/デバイスの分担 | src/core/layer.cpp, src/program/generate.cpp |
| [01 メモリ](01_memory.md) | SSD → RAM → VRAM → SMEM の 4 段、expert cache、PCIe リング | src/core/{vmm,expert_cache,pinned}.cu |
| [02 量子化](02_quant.md) | S2/S4/S8、IQ 系、Q8_0/Q8_K 活性のブロックレイアウト | s_gemv.hpp, iq_kernels.hpp, docs/pack-format.md |
| [03 Decode GEMV](03_decode_gemv.md) | 重み 31.64GiB を読むホットパス | s_gemv.cu ★ 全注釈付き |
| [04 MoE](04_moe.md) | router → CPU プール → combine、shared expert | s2_expert_grouped.cu, shared_expert.cu |
| [05 Attention / GDN](05_attention_gdn.md) | QSA(indexer+topk+attend)、KV 圧縮、DeltaNet | qsa*.cu, kv_*.cu, fused_gdn.cu |
| [06 Prefill](06_prefill.md) | int8 テンソルコア MoE、cuBLAS GEMM、GDN 再帰 | src/prefill/*.cu |
| [07 その他](07_misc.md) | sampler、PLE、GR、verify、elementwise | sampler.cu, ple.cu, fused_gr.cu |
| [08 ファイル目録](08_catalog.md) | 46+9 ファイルの役割・行数・起動構成 | 全ファイル |

図はすべて `diagrams/png/` の PNG（[index.html](../index.html) の「図」節からも辿れる）。
1 トークンの流れは、まずこの 1 枚を見よ。

> 図: [00 1 トークン・1 レイヤー](../diagrams/token-pipeline.svg)

---

## 1. まず結論: このエンジンの設計思想

Strata は「**GPU に載らないものは、載せないまま計算する**」という一点に賭けたエンジンである。
125B のモデルは 2bit 量子化しても 38GiB あり、24GB の VRAM には収まらない。普通の実装は
「量子化を強める」「層を CPU に逃がす（オフロード）」のどちらかを選ぶ。Strata はそのどちらでもなく、

> **重みは全部 RAM に常駐させ、GPU が持っているのは“注意力とルータと少数のホット expert”だけ。
> expert の計算は、GPU が Attention を計算している間に CPU が自分でやる。**

を選んだ。そのためエンジンの本体は「GEMV を速くする」ことではなく、
**GPU と CPU をどう重ねるか（スケジューリングとドアベル）** に置かれている。
.cu を読むときも、常に「これは誰と並列に走るのか」を意識すると読みやすい。

---

## 2. 3 つの実行モード

```
                 ┌──────────── decode（1トークンずつ） ────────────┐
 入力トークン →  │ router → expert選択 → GEMV(MoE) → attention → head │ → 1トークン
                 └────────────────────────────────────────────────────┘

                 ┌───────── verify（MTP 投機の検証、最大4トークン） ──┐
 下書き 3 個 →   │ 同じ重みで 4 トークン分を一括 forward → 受理/棄却   │
                 └────────────────────────────────────────────────────┘

                 ┌───────── prefill（プロンプト、最大8192トークン） ──┐
 長い入力 →      │ GEMM（テンソルコア）でまとめて、expertはPCIeで流す  │
                 └────────────────────────────────────────────────────┘
```

- **decode**: バッチ 1。GEMV（行列×ベクトル）。メモリ帯域律速なので、カーネルは
  「重み 1 バイトを何回の命令で消費するか」を削る競技になる → 第 3 章。
- **verify**: 投機的デコードの検証。バッチは最大 4 だが、**算術は decode とビット一致**でなければ
  ならない（`verify_kernels.cu` の冒頭コメント）。ここが本エンジンで一番「職人芸」の箇所。
- **prefill**: バッチ数千。GEMV ではなく GEMM なので、int8 テンソルコアで
  量子化重みを直接掛ける（`moe_fused.cu`）→ 第 6 章。

---

## 3. レイヤの解剖

Qwen3.8-Flash-Next の 1 層は、大きく 3 つのブロックからなる。

```
   x (残差ストリーム)
   │
   ├─▶ [ GR (gated residual) ] ── 正規化 + ゲート付き残差書き込み ──┐
   │                                                              │
   ├─▶ [ Mixer ]  ← 層によって 2 種類                        (残差に加算)
   │      ├─ GDN 層 : DeltaNet（線形注意、状態 S×h_v×S を持つ）
   │      └─ QSA 層 : Qwen Sparse Attention（indexer で top-k を選ぶ疎な注意）
   │
   └─▶ [ MoE ]
          ├─ router: 24,576 expert から top-10 を選ぶ（BF16 必須）
          ├─ routed experts: 10 個（GPU 常駐 or CPU プール）
          └─ shared expert: 全トークン共通の 1 個（常に GPU）
```

### GR（gated residual）が地味に重い

`fused_gr.cu` は **1,991 行**あり、単体ファイルとしては 3 番目に大きい。それだけ
「残差の書き込み方」に神経を使っている。理由は、MoE の計算を CPU に投げる間に
残差ストリームを壊してはいけないからで、**pre 側で GR を書き、post 側で読み戻す**という
2 段階のプロトコルになっている。

### 分割グラフ（このエンジンの心臓）

```
block_layer_pre[l]                          block_layer_post[l]
┌────────────────────────────┐              ┌───────────────────────────┐
│ attn(QSA/GDN) → GR write   │              │ moe_finish → GR write     │
│ → ffn gr_read → moe_route  │              │                           │
│        │                   │              │                           │
│        └── ドアベル ───────┼──▶ ホスト ──▶│                           │
└────────────────────────────┘   CPU プール  └───────────────────────────┘
        GPU が走る                       CPU が走る            GPU が走る
```

`moe_route` の最後で **ドアベル**（mapped pinned memory にシーケンス番号を書くカーネル）を鳴らすと、
ホスト側のスレッドがそれを待っていて、CPU プールで「GPU に載っていない expert」を計算し始める。
GPU はその間、`shared_expert`（既定で pre 側に前倒し = `layer_set_shared_early`）を計算する。

> **落とし穴（round 199 の記録）**: `cudaEventRecord` は CUDA Graph のキャプチャ中では
> **黙って落ちる**。だから同期はイベントではなく、明示的にカーネルが書くドアベルにしてある。
> これは `elementwise.cu` の `doorbell_ring_kernel` にある。


---

## 4. ホストとデバイスの分担

| 担当 | やること | 代表ファイル |
|---|---|---|
| GPU | attention, router, shared expert, VRAM 常駐 expert, KV, head, sampler | `src/kernels/cuda/*` |
| CPU | **GPU に載っていない expert の GEMV**（AVX-512 / AVX2、ggml の i-quant カーネル） | `ref/` と `bench/micro/cpu_s2.cpp` |
| ホスト制御 | CUDA Graph の組み立て、ドアベル待ち、expert cache の更新、PCIe 転送の発行 | `src/core/layer.cpp`（1,510行）, `src/program/generate.cpp`（13,110行） |

`layer.cpp` の `gemv_quantized()` はこの本で一番よく出てくる分岐なので、最初に要約を置く:

```cpp
// 重みの形式で 4 通りに分かれる（層の中で混在する）
if (native_data)                        // ネイティブ(i-quant)パック
    native_quantize_q8_1(...); native_mmvq(type, ...);   // llama.cpp の MMVQ を再利用
else if (form.code_bits == 2 && act_is_q8_0)
    quantize_q8_0(...); s2_gemv_q8(...);                 // Q2_0 専用の高速路
else if (wants_q8k)
    quantize_q8_K(...); s_gemv_q8k_split(...);           // K-quant / IQ4_XS
else
    quantize_q8_0(...); s_gemv_q8_0_split(...);          // ffn_down_shexp (n_in=640)
```

この 4 分岐を取り違えると、**同じ forward の中で expert ごとに違う数値**が出る。
`quantize_act.cu` の冒頭には「Q8_0 と Q8_K を取り違えると GEMV 誤差 0.6〜1.4%」と記録されている。
→ 第 2 章・第 3 章で詳述。

---

## 5. 数値契約という文化

このリポジトリのコメントは異常に詳しい。理由はコードの冒頭に書いてある:

> 「**算術は転記する（transcribed）のであって、思い出す（recalled）ものではない**」

つまり llama.cpp のカーネルとビット単位で一致させることを目的にしていて、一致しない場合は
「どちらが正しいか」ではなく「**どこで丸めが違ったか**」を特定してコメントに残す運用になっている。
代表例:

- `quantize_act.cu`: `d32 = amax/127` は FP32 で計算し、**量子化後の整数は d32 で割る**。
  保存される fp16 丸め値で割ると 80 ブロック中 18 ブロックが ggml のバイト列と食い違った。
- `s_gemv.cu`: offset（Q4_K の min）は **重みに足してから**活性と掛ける。
  `acc += code*scale*x + offset*x` と書くと数学的には同じだが丸めが変わる。
- `rope.cu`: ペアリングは **NEOX**（`i` と `i + n_rot/2`）。隣接ペア（`2i, 2i+1`）と間違えると
  「形は正しいが中身がスクランブルされた出力」が出る。
- `shared_expert.cu`: SiLU は **gate 側**。up 側にかけても形は合ってしまう。

だからこの本も「結論」より「**なぜその順序でなければならないか**」を優先して書く。

---

## 6. どこから読むか（おすすめ順）

1. `docs/pack-format.md`（パックの物理レイアウト、S2 の 2bit 配置）
2. **`src/kernels/cuda/s_gemv.cu`**（[第3章](03_decode_gemv.md) に全注釈）— エンジンの 8 割はここ
3. `src/kernels/cuda/elementwise.cu` のドアベル 4 関数 — 並列の起点
4. `src/core/layer.cpp` の `gemv_quantized` と `block_layer_pre/post` — 呼び出し順
5. `src/prefill/moe_fused.cu` 冒頭 60 行 — 「魔法の float」トリックの説明
6. `src/kernels/cuda/sampler.cu` 冒頭 — サンプラ順序の落とし穴

---

## 7. 用語

| 語 | 意味 |
|---|---|
| S2 / S4 / S8 | Strata 独自の「コード幅」表現。S2 = 2bit コード + 64要素ごとの FP32 スケール（Q2_0 の正規形） |
| code_bits | 1 要素あたりのビット数（2/4/8）。13 種類の量子化形式が 3 幅に潰れる |
| act_kind | 活性の量子化形式。Q8_0(34B/32) と Q8_K(292B/256) の 2 種。**形式からは導出できない**ので明示 |
| ドアベル | カーネルが書く同期用の番号。CUDA Graph 内でイベントが使えないための代替 |
| CPU プール | GPU に載っていない expert を計算するホスト側スレッド群 |
| MTP | Multi-Token Prediction。投機の下書き層 |
| PLE | 28.8GB の n-gram テーブル。SSD から直接読む |

---

*次: [01 メモリ階層](01_memory.md) — SSD から SMEM までの 4 段を、サイズと帯域の実数で追う。*

