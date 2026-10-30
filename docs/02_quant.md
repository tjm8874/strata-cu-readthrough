# 02. 量子化フォーマット — 13 形式が 3 つの「コード幅」になる

> 図: [02 量子化ブロック — S2 の 20 バイトと decode / prefill の内側](../diagrams/quant-block.svg)

## 1. なぜ「S2/S4/S8」なのか

Qwen3.8-Flash-Next のパックには、量子化形式が **13 種類**混在している。
層ごとに、テンソルごとに違う（例: `attn_qkv` は IQ4_XS が 13 層、Q3_K が 18 層、Q2_0 が 1 層、Q4_K が 4 層）。

これをカーネル側で 13 分岐すると、内側ループに switch が入る。Strata の答えは
**「コード幅（1 要素あたりのビット数）だけをコンパイル時定数にし、残りは実行時引数にする」**。

```cpp
// include/strata/kernels/s_gemv.hpp
struct SForm {
    int   code_bits;     // 2 / 4 / 8   ← これだけテンプレート引数
    int   code_bias;     // 整数域で足すバイアス（Q2_0 は -1）
    int   group_elems;   // スケール 1 個が受け持つ要素数（16 / 32 / 64）
    Codebook codebook;   // Affine | Iq4Nl
    bool  has_offset;    // Q4_K / Q5_K の min プレーン
    int   act_kind;      // 0 = Q8_0, 1 = Q8_K  ← 形式から導出できない
};
```

3 幅 × コードブック 2 種 × offset 有無 = 実質 12 通りだが、カーネルは
`template <int CODE_BITS>` の 3 実体で済む。残りはカーネル引数。

**`act_kind` だけは導出できない**のが罠。`Q5_0` と `Q5_K` はコード幅が同じでも
活性の要求が違い、`IQ4_NL` と `IQ4_XS` も同様。だから manifest が明示する。

---

## 2. S2（= Q2_0 の正規形）のビット配置

**全 38 GiB のうち 31.64 GiB が S2**。したがって S2 のカーネルが勝敗を決める。

```
 1 グループ = 64 要素
 ┌───────────────────────────────┬──────────────┐
 │ codes: 64 × 2bit = 16 バイト  │ scale: FP32  │
 └───────────────────────────────┴──────────────┘
   バイト i に 要素 4i, 4i+1, 4i+2, 4i+3 が入る（LSB ファースト）
   bit 位置:  要素 4i+k は ビット [2k, 2k+2)

   復号:  w = (code + bias) * scale      bias = -1 は【整数域で】足す
```

「bias を整数域で足す」が重要。`(float)code * scale - scale` と書くと丸めが 1 回増える。
`docs/pack-format.md` §3.1 に、この点が「正規形（canonical form）」として定義されている。

### テンソルの格納順

GGUF 順 `[n_in, n_out]`、**n_in が連続**。つまり出力行 `o` は
`n_in / group_elems` グループの連続した並びで、行の先頭は必ずグループ境界。
だからカーネルは「行ポインタ + グループ走査」だけで済む（転置不要）。

---

## 3. 活性の量子化 — Q8_0 / Q8_1 / Q8_K

ggml の GEMV は「重み × 活性」を整数ドット積でやるために、**活性を重みの `vec_dot_type` に
変換してから**掛ける（`ref/quant.py` の `apply_src1`）。Strata も同じ規則に従う。

| 形式 | ブロック | 中身 | 使う場所 |
|---|---|---|---|
| Q8_0 | 32 要素 / **34 B** | fp16 d + int8 qs[32] | レガシー形式（Q4_0/Q5_0/IQ4_NL…）の `vec_dot_type` |
| Q8_1 | 32 要素 / **36 B** | fp16 d + fp16 s + int8 qs[32] | llama.cpp MMVQ 系（`native_mmvq`） |
| Q8_K | 256 要素 / **292 B** | f32 d + int8 qs[256] + int16 bsums[16] | K-quant / IQ4_XS |

`bsums` は Strata では読まない（ggml の AVX2 用）が、**ストライドに含まれる**ので
`quantize_q8_K` の出力をそのまま渡せる。

### 構造的な制約

`ffn_down_shexp` は **n_in = 640**。256 の倍数ではないので **Q8_K は物理的に不可能**。
だからこのテンソル専用に `s_gemv_q8_0_split` がある（第 3 章）。
「Q8_K が未実装」ではなく「この形では実装できない」のが正しい理解。

### 罠: d32 の丸め

`quantize_act.cu` の冒頭:

> `d32 = amax / 127.0f` は **FP32** で計算し、量子化後の整数は **d32 で割る**。
> 保存される fp16 丸め値で割ると、ggml のバイト列と **80 ブロック中 18 ブロック**が食い違った。
> しかも Q2_0 expert は Q8_0 に変換されるので、誤差は 48 個の MoE ブロック全部に乗り、
> **end-to-end の KL を悪化させた**。

「保存する値」と「計算に使う値」を混同すると、目に見えない形でモデル全体が劣化する典型例。

---

## 4. i-quant のブロックサイズ

`moe_fused_iq.cu` が扱う形式（`static_assert` でサイズを固定）:

| 形式 | ブロックバイト | 備考 |
|---|---|---|
| IQ2_XXS | 66 | コードブック + 符号ビット、8 要素単位のスケール |
| IQ2_XS | 74 | |
| IQ2_S | 82 | |
| IQ3_XXS | 98 | |
| IQ3_S | 110 | |
| IQ4_XS | 136 | 32 要素ごとに scale |
| Q2_0 | 18 | codes 16B + fp16 d |

※ 2 バイト境界に整列しているので 16bit ロードで読める、という前提が
`moe_fused_iq.cu` のロード段の設計に効いている。

---

## 5. 整数ドット積を float にする「魔法の float」

`moe_fused.cu`（prefill の Q2_0 経路）の核心。int8 テンソルコアは int32 を返すので、
これを float に変換すると 1 命令（I2F, 高コスト）かかる。Strata はこれを消す:

```cpp
// 1.5 * 2^23 = 0x4B400000。この仮数部に int をそのまま足すと、
// 指数部が変わらない範囲（|dot| < 2^22）で float の値が 1.5*2^23 + dot になる
as_float(0x4B400000 + dot) - 1.5f*2^23  ==  (float)dot   // 丸めなし・I2F なし
```

Q2_0 の場合、重みは `w = d_w*(q-1)`、活性は `x = d_x*a` なので

```
Σ w x = d_w * d_x * (Σ q a - Σ a)
```

`Σa` は活性ブロックに保存済み（`c = -(1.5*2^23 + Σa)`）。したがって

```cpp
acc += d_w * (d_x0 * t0 + d_x1 * t1);   // t = dot - Σa, 整数 2 回 + float 3 回
```

で 1 ブロック分が終わる。**変換命令ゼロ**。この発想は `moe_fused_iq.cu` にも展開されていて、
そちらは `MAGICF` という名前で同じトリックを使う。

---

## 6. まとめ表: どこで何が量子化されるか

| 段階 | 形式 | 実装 |
|---|---|---|
| パック（オフライン） | S2/S4/S8, IQ 系, K-quant | `docs/pack-format.md` |
| 活性（decode） | Q8_0 / Q8_K / Q8_1 | `quantize_act.cu`, `native_mmvq.cu` |
| 重み読み出し（decode） | そのまま（復号しながら掛ける） | `s_gemv.cu`, `iq_kernels.cu` |
| 重み（prefill） | int8 に展開してテンソルコアへ | `moe_fused.cu`, `moe_fused_iq.cu` |
| KV | f16 → q8 → q4（Hadamard 回転） | `kv_q8.cu`, `kv_q4.cu`, `cvec.cu` |

---

*次: [03 Decode GEMV](03_decode_gemv.md) — 31.64 GiB を読むカーネルの全注釈。*

