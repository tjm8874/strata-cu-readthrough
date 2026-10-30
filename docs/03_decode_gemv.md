# 03. Decode GEMV — 31.64 GiB を 1 トークンで読む

> 対象: `src/kernels/cuda/s_gemv.cu`（694 行、**全行注釈**）
> 関連: `s2_gemv*.cu`, `iq_kernels.cu`, `native_mmvq.cu`, `quantize_act.cu`

decode の 1 トークンは、モデルの重み（38 GiB のうち VRAM 常駐分 + top-10 expert）を
**1 回ずつ**読む。したがって decode は完全な **メモリ帯域律速**であり、
カーネルの仕事は「1 バイトの重みを、何命令・何回のメモリアクセスで消費するか」を最小化すること。

`s_gemv.cu` には 4 つの実装が同居している。この章はその全部を、コメントに書かれた
**実測値と失敗談**ごと読む。

| 実装 | 行 | 用途 | 並列度 |
|---|---|---|---|
| `s_gemv_kernel` | 301–340 | 素直な参照実装（1 thread = 1 出力行） | n_out |
| `s_gemv_split_kernel` | 400–474 | `attn_output` / `shared_expert` 用 | n_out × TPR |
| `s_gemv_q8k_kernel` | 90–127 | Q8_K 活性（K-quant / IQ4_XS） | n_out |
| `s_gemv_q8_split_kernel` | 142–299 | **主力**。Q8_K / Q8_0 両対応、1 warp/行 | n_out × 8 |

---

## 1. 16 バイトのコードブックが 2.12 倍を稼ぐ

ファイル冒頭、19 行目からの長いコメントは IQ4_NL のコードブックの話。

```cpp
__constant__ signed char kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                        1,   13,   25,  38,  53,  69,  89,  113};
```

> **IT WAS IN `__constant__` FOR THE WRONG REASON, AND IT COST 2.12x.**
> 旧コメントは「全スレッドが読むし、ラウンチ全体で同じ 16 値だから」と constant メモリを正当化していた。
> その理由付けは**まったく逆**である。constant メモリが速いのはアクセスが **UNIFORM**
> （全レーンが同じアドレス）のときで、非線形コードブックは**レーンごとに違う index** を読む。
> それは constant が最も苦手なパターンだ。

測定（`attn_qkv` IQ4_XS 10240×2560、コードブックを Affine に固定してメモリ量と命令数を揃え、
テーブル読みだけを消した比較）:

```
    IQ4NL  0.2032 ms
    Affine 0.0960 ms      比 2.12x
```

修正は「shared に 16 バイト置く」。ディバージェントな 16 エントリ読みでも、
shared なら最大 4-way バンクコンフリクトで済む。

```cpp
template <int CODE_BITS>
__device__ __forceinline__ float decode_tbl(int code, int bias, int codebook, const signed char* tbl) {
    if (codebook == (int) Codebook::Iq4Nl) return (float) tbl[code & 0x0F];
    return (float) (code + bias);   // bias は【コード（整数）】に足す
}
```

> **教訓**: 「全員が読むから constant」は間違い。正しい基準は「全員が**同じ**アドレスを読むか」。

### バリアの位置という古典的バグ

各カーネルの先頭 3 行は必ずこの形:

```cpp
__shared__ signed char s_iq4nl[16];
if (threadIdx.x < 16) s_iq4nl[threadIdx.x] = kIq4Nl[threadIdx.x];
__syncthreads();
const long long o = ...;
if (o >= n_out) return;          // ← 早期リターンはバリアの【後】
```

コメントに「ロードとバリアは早期リターンより**前**に置く。`o` がスレッド単位のカーネルでは、
先に return したスレッドが、他のスレッドがまだ待っているバリアに到達しなくなる」とある。
これは `s_gemv_q8_split_kernel` で実際に`n_out % 8 != 0` のとき未定義動作になっていた
（パックの形が全部 8 の倍数だったので表面化しなかった）。

---

## 2. 参照実装: `s_gemv_kernel`（301–340 行）

```cpp
constexpr int PER_BYTE = 8 / CODE_BITS;          // S2:4, S4:2, S8:1 個/バイト
const long long n_groups = n_in / group_elems;
const uint8_t* c   = codes + o * (n_in / PER_BYTE);
const float*   s   = scales + o * n_groups;
const float*   off = has_offset ? offset + o * n_groups : nullptr;

for (long long g = 0; g < n_groups; ++g) {
    const float d = s[g], b = off ? off[g] : 0.0f;
    for (int j = 0; j < group_elems; ++j) {
        const long long i = g * group_elems + j;
        const int code = (c[i / PER_BYTE] >> ((int)(i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
        const float w = decode_tbl<CODE_BITS>(code, bias, codebook, s_iq4nl) * d + b;
        acc += w * __half2float(__ushort_as_half(x[i]));
    }
}
```

### ここで 1 つ、将来 3 時間を溶かす罠が書いてある

> offset は**重み**に属するので、活性と掛ける**前に**復号値へ足す:
> `w = code*scale + offset; acc += w * x`
> `acc += code*scale*x + offset*x` と書くと数学的に等しい式だが**丸めが違う**
> （乗算 2 回 + 加算 1 回 vs 乗算 1 回 + 加算 1 回）。
> offset の無い形式では区別できないので、**Q4_K を追加するまで間違ったまま**だった。

これは「テストが通る = 正しい」ではないことの見本。offset を持つ形式（Q4_K/Q5_K）を
パックに足した瞬間に初めて差が出る。

---

## 3. 主力: `s_gemv_q8_split_kernel`（142–299 行）

> 図: [02 量子化ブロック — S2 の 20 バイトと decode の内側](../diagrams/quant-block.svg)

1 warp = 1 出力行、レーンが縮約軸をストライドする。**16 要素 / レーン / イテレーション**。

### 3.1 なぜ「quad（4 要素）」から「オクテット（16 要素）」へ広げたか

最初のバージョンは 1 要素ずつ、`i = lane, lane+32, ...` で読んでいた:
**要素ごとにコードロード・シフト・スケールロード**。しかし

- 4 連続要素は 1 つのコードワードを共有する（S2 なら 1 バイト、S4 なら 2 バイト、S8 なら 4 バイト）
- どの形式でもグループサイズは 4 の倍数なので、quad は 1 グループに収まる

そこで quad にすると「コード 1 回・スケール 1 回・シフト 1 回」で 4 要素。
さらに **QE = 16**（4 quad）に拡張し、2 つのコードワードを「消費する前に発行」して
バイト・イン・フライトを増やす。

```cpp
constexpr int QE = 16;
constexpr int QW = 4 * CODE_BITS / 8;   // 4 コードワードのバイト数: S2=1, S4=2, S8=4
constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
long long i = (long long) lane * QE;
for (; i + QE <= n_in; i += 32 * QE) { ... }
```

> **コメントの自己申告**: 「`QW` は 4 コードワードあたりのバイト数であって、
> オクテット全体の幅ではない。最初の版はこれを取り違え、S4 で uint32 を読み、S8 で 1 バイトを読んだ。
> `s_gemv_q8k_parity` がまさにそこで落ちた。**それがこのテストの存在理由**」。

### 3.2 除算をシフトにする（1 要素あたり 100 サイクルの削減）

グループ添字は本来 `e / group_elems`。**64bit 整数除算は GPU ではソフトウェアルーチンで約 100 サイクル**。
これがエンジンで一番熱いループの中で要素ごとに走っていた。

```cpp
const long long g = i >> group_shift;   // group_elems は必ず 2 の冪（16/32/64）
```

ホスト側は「2 の冪でなければ**静かに間違える**のではなく exit する」:

```cpp
if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
    std::fprintf(stderr, "s_gemv_split: group_elems must be a power of two, got %d\n", form.group_elems);
    std::exit(1);
}
```

### 3.3 活性ブロックの巻き上げ — ここが本当の勝ち

`q8k_at(x, i + k)` は呼び出しごとに `(i+k)/256` と `(i+k)%256` を計算する。
つまり **要素ごとに 64bit 除算 + 剰余**、1 レーンイテレーションで 16 回。

しかし `QE = 16`、ブロックは 256（Q8_K）または 32（Q8_0）、ループの増分は `32*QE` なので、
`i % 256` も `i % 32` も常に 16 の倍数 → **16 要素は必ず 1 ブロックに収まる**。
よって除算は 1 回でよい:

```cpp
const int blk_elems = Q8K ? Q8K_BLOCK_ELEMS : Q8_0_BLOCK_ELEMS;
const uint8_t* xb = x + (i / blk_elems) * (Q8K ? Q8K_BLOCK_BYTES : Q8_0_BLOCK_BYTES);
const int xi = (int)(i % blk_elems);
const float xd = Q8K ? __ldg((const float*) xb)
                     : __half2float(__ushort_as_half(*(const uint16_t*) xb));
const int8_t* xq = (const int8_t*)(xb + (Q8K ? 4 : 2)) + xi;   // 16 連続 int8 ロード
```

> 「これは近似ではなく**厳密**。`xi + QE <= blk_elems` が到達可能な全ての `xi` で成り立つので、
> レーンイテレーションがブロック境界をまたぐことはない」。
> 端数は末尾の 1 回きりのテールループが担当し、そちらは元のアクセサを使う。

**この 1 点だけで 16 個の 64bit 除算が消える**。メモリアクセスを減らす話より効くことがある、
というのがこのカーネルの一番の学び。


### 3.4 16 個のアキュムレータと、順序が変わることの代償

```cpp
float acc0..acc15;   // 16 本の独立した依存チェーン
...
acc0 += w0 * (xd * (float) xq[0]);
...
float acc = (((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))) +
            (((acc8 + acc9) + (acc10 + acc11)) + ((acc12 + acc13) + (acc14 + acc15)));
for (int step = 16; step > 0; step >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, step);
if (lane == 0) y[o] = acc;
```

理由はコメントに書いてある: 「このカーネルは FP32 ピークの 0.4%、帯域の 2.1% で走っている。
どちらも律速ではない。律速は**依存チェーン**で、`acc += ...` が 1 本だと
FMA のレイテンシ（4–6 サイクル）が直列化し、SM が遊ぶ」。

**代償**: 加算の順序が変わるのでビット一致しない。だから
`s_gemv_parity` / `s_gemv_q8k_parity` は「ナイーブ版との差」を許容誤差つきで比較する。
「速くする → 順序が変わる → パリティテストで守る」がこのエンジンの定型。

### 3.5 ワープ縮約

`__shfl_down_sync` で 31→1 レーンへ。`lane == 0` だけが書く。
アトミックを使わないのは**決定性**のため（コメント: 「レースはテストを
“間違って” ではなく “不安定に” する。それが最悪の失敗の仕方」）。

---

## 4. `s_gemv_split_kernel`（400–474 行）— 出力幅が狭いときの救世主

`s_gemv_kernel` の並列度は `n_out`（出力行数）。ところが expert の gate/up は
`[2560 × 640]` で、**640 スレッドでは 48 SM を埋められない**。
コメントに「gate/up [2560x640] は [640x2560] の 4.6 倍遅かった」と実測がある
（`s_gemv` は行方向に並列なので、行数が少ないと並列度が落ちる）。

そこで「**1 ブロック = 1 出力行、`threads_per_row` 個のスレッドで縮約**」:

```cpp
template <int CODE_BITS>
__global__ void s_gemv_split_kernel(...) {
    extern __shared__ float partial[];
    ...
    const long long o = blockIdx.x;          // ← ブロックが行に対応
    const int tid = threadIdx.x;
    ...
    constexpr int QE = 4;                    // 1 スレッド 4 要素
    constexpr int QB = QE * CODE_BITS / 8;
    float acc0, acc1, acc2, acc3;            // 4 本のチェーン
    long long i = (long long) tid * QE;
    for (; i + QE <= n_in; i += (long long) threads_per_row * QE) {
        ...
        const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(x + i));      // 4 half を 8B で
        const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(x + i + 2));
        ...
    }
    partial[tid] = (acc0 + acc1) + (acc2 + acc3);
    __syncthreads();
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {     // 固定ツリー
        if (tid < step) partial[tid] += partial[tid + step];
        __syncthreads();
    }
    if (tid == 0) y[o] = partial[0];
}
```

- **固定ツリー縮約、アトミックなし** → 決定性。
- `TPR = 32` が実測の最適（64 にすると 11.75 vs 11.99 tok/s）。
  `threads_per_row` は 2 の冪 1..1024 を強制。
- 4 half を 1 回の 8 バイトロードで読む（`i` は 4 の倍数なので 8 バイト整列）。

> **どちらのカーネルを使うか**は呼び出し側（`layer.cpp`）の判断。
> 出力幅が広い（10240 行など）なら `s_gemv_split` で十分、
> 640 行のような狭い行列は `s_gemv_split`（ブロック/行）でないと埋まらない。

---

## 5. なぜ Q8_0 専用カーネルが「構造的に」必要か（647–692 行）

```cpp
// THE ACTIVATION IS block_q8_0, 32 elements per block, so n_in must be a multiple of 32 -
// which is a WEAKER requirement than Q8_K's 256 and is exactly why this kernel has to exist:
// ffn_down_shexp has n_in = 640, a multiple of 32 that is NOT a multiple of 256.
if (n_in % Q8_0_BLOCK_ELEMS != 0) { ... exit(1); }
```

そして Q8_K 側は 2bit を**明示的に拒否**する:

```cpp
default:
    // 2-bit S2 never has a Q8_K activation: its vec_dot_type is Q8_0.
    // Refusing is better than running a kernel that would be numerically wrong in a way
    // the caller cannot see.
    std::fprintf(stderr, "s_gemv_q8k: code_bits %d has no Q8_K contract ...");
    std::exit(1);
```

> **設計原則**: 「間違った数値を静かに出す」より「起動時に落ちる」。
> このエンジンは `docs/activation-contract.md` という 1 枚の表で契約を管理している。

---

## 6. 負けた実装も残す — `s2_gemv_fast.cu` と `#pragma unroll`

`s2_gemv_fast.cu`（153 行）は名前と裏腹に**速くない**。試した 2 つのレバー:

1. 活性 `x` を shared にステージする
2. `__constant__ float c_codes[256][4]` の LUT（1 バイト→4 コードの展開表）

両方とも中立〜悪化。しかも**実行ごとのばらつきが約 20%** あって（verify が 6 コアの 97% を
握っているため）、効果量がノイズ床より小さい。だから「差が無い」と結論できないまま保留になった。

`s_gemv_q8_split_kernel` 側にも同種の記録がある:

```cpp
// **`#pragma unroll 4` WAS TRIED HERE AND DID NOTHING.**
// 測定: n_out 10240 で 184.6 → 183.9 GB/s。パリティは両方クリーン。
// 仮説 L93（行あたり約 6ns はバイト・イン・フライト限界）は棄却されないが、
// 「命令スケジューリングが律速」は棄却された。
// Removed rather than kept, because a hint that does not change the timing is a claim that is not true.
```

**「効かなかった最適化は消す」** — これも設計文化。コメントに効果が無いと書いてある
ヒントは、嘘をついているのと同じだから。

---

## 7. 数字で見る decode の現実

| 指標 | 値 | 出典 |
|---|---|---|
| S2 GEMV の重み処理レート | 66 G weights/s = 0.55 w/cycle/SM | `s2_gemv_quads.cu` コメント（L4 測定） |
| 同カーネルの FP32 ピーク比 | 0.4% | 同上（＝計算律速ではない） |
| 帯域比 | 2.1% | 同上（＝帯域でもない） |
| 行あたり時間 | 約 6 ns/出力行 | `s_gemv.cu` の L93 参照 |
| `__constant__` → shared | 2.12x | `s_gemv.cu` 19–31 行 |
| TPR 32 vs 64 | 11.99 vs 11.75 tok/s | ホスト側測定 |

計算でも帯域でもないなら何が律速か。候補は「命令発行」か「バイト・イン・フライト」。
`s2_gemv_quads.cu` の冒頭はこの 2 つを明示的に区別して実験を設計している
（13 命令/重み × 4/cycle 上限 = 7.2 命令/cycle 要求、という見積り付き）。

---

## 8. この章のチェックリスト（読むときの観点）

1. **コードワードは何要素を共有するか** → 1 回のロードで何要素処理できるかが決まる
2. **グループ添字は除算かシフトか** → 熱いループなら 100 サイクル/要素の差
3. **アキュムレータは何本か** → FMA レイテンシを隠せているか
4. **バリアは早期リターンより前か** → 未定義動作の定番
5. **活性のブロック境界をまたがないことは証明できるか** → 除算を巻き上げられるか
6. **offset は重みに足してから掛けているか** → 丸めが変わる
7. **その最適化は本当に効いたか** → 効かないなら消す（`s2_gemv_fast` の教訓）

---

*次: [04 MoE](04_moe.md) — router が選んだ 10 個を、GPU と CPU で分担して計算する。*

