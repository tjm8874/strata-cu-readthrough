# 01. メモリ階層 — SSD から SMEM まで

> 図: [01 メモリ階層 — SSD → RAM → VRAM → SMEM](../diagrams/memory-hierarchy.svg)

125B/2bit のパックは **38 GiB**、PLE テーブルは **28.8 GB**。どちらも VRAM には入らない。
この章は「どのバイトが、どの階層に、どういう経路で移動するか」を追う。

---

## 1. 4 つの階層

```
┌──────────────────────────────────────────────────────────────────────┐
│ SSD / NVMe                                            38 GiB + 28.8GB│
│   ├─ モデルパック (GGUF 相当, mmap しない)                           │
│   └─ PLE テーブル (28.8GB, トークンごとに数行だけ unbuffered read)   │
└───────────────┬──────────────────────────────────────────────────────┘
                │ 起動時に一度だけ / PLE は毎トークン数行
┌───────────────▼──────────────────────────────────────────────────────┐
│ システム RAM (pinned / mlock)                          ≒ 40–70 GiB    │
│   ├─ 24,576 expert 全部（常駐。追い出さない）                         │
│   ├─ PCIe リング用バッファ（BATCHED_DMA）                            │
│   ├─ ドアベル（mapped pinned memory, GPU から書ける）                │
│   └─ KV のホスト側ミラー（64K 超の cold 部分）                       │
└───────────────┬──────────────────────────────────────────────────────┘
                │ PCIe Gen5 x16 (H2D / D2H, DMA)
┌───────────────▼──────────────────────────────────────────────────────┐
│ VRAM 12–24 GiB                                                       │
│   ├─ attention / GDN の重み（層ごとに常駐）                          │
│   ├─ GR 重み、router、shared expert、output head、MTP 層            │
│   ├─ KV cache（ホット部分のみ）                                      │
│   ├─ expert cache（適応型。約 700 expert / GiB）                     │
│   └─ 作業バッファ（活性の量子化イメージ、combine 用）                │
└───────────────┬──────────────────────────────────────────────────────┘
                │ カーネル内ロード
┌───────────────▼──────────────────────────────────────────────────────┐
│ SMEM / レジスタ (1 SM ≈ 228 KB SMEM, 65536 regs)                     │
│   ├─ iq4nl コードブック 16B（全 GEMV カーネル共通の定番）            │
│   ├─ attention のスコア配列（1 head あたり最大 8 KB）                │
│   └─ prefill のタイル（cp.async で二重バッファ）                     │
└──────────────────────────────────────────────────────────────────────┘
```

### 帯域で見ると何が効くか

decode の 1 トークンは、**48 層 × (attention 重み + top-10 expert)** を読む。
top-10 expert は 1 層あたり 10/24,576 だが、RAM 側は 10 expert 分を CPU が読む。
つまり 1 トークンあたりの実効読み出しは「数十 GB/s 級の RAM 帯域 + VRAM 帯域」の合成で、
**どちらのパスも遊ばせないこと**が速度の全てになる。これが第 0 章の「分割グラフ」の理由。

---

## 2. expert の 3 つの居場所と移動

> 図: [04 expert の振り分け — router → 3 経路 → combine](../diagrams/expert-flow.svg)

top-10 に選ばれた expert は、その瞬間に 3 つのどれかにいる:

| 居場所 | 誰が計算する | コスト |
|---|---|---|
| **A. VRAM 常駐**（expert cache ヒット） | GPU（`s2_expert_grouped.cu`） | 最速。VRAM 帯域のみ |
| **B. RAM（キャッシュミス）** | CPU プール（AVX-512/AVX2） | PCIe 転送ゼロ。CPU コア数に律速 |
| **C. ピア GPU**（2 枚目） | ピアの GPU | P2P 転送。`SECOND_GPU.md`, `peer_experts.cpp` |

**B が本エンジンの発明**である。普通のオフロードは「expert を GPU に持ってきて計算する」ので
PCIe がボトルネックになる。Strata は「**持ってこない。CPU で計算して結果だけ返す**」。
そのため VRAM 側は attention に専念でき、PCIe は prefill のリングに空けられる。

### cache の学習

`src/core/expert_cache.cpp` が「どの expert がよく選ばれるか」を学習し、VRAM の空きに応じて
約 700 expert/GiB の密度で常駐させる。学習の材料は routing の履歴で、**層ごとに別々**に持つ
（層によって偏りが全く違うため）。

### 移動の実装

- `src/core/pinned.cu`（800 行）: ページロックされたホストバッファの管理。
  `cudaHostAlloc` + マップ（GPU から直接書ける領域）。ドアベルはここ。
- `src/core/vmm.cu`: 仮想メモリ管理。VRAM の断片化を避けるため、
  あらかじめ確保した領域を「見かけ上」差し替える（物理コピーを減らす）。
- `src/core/remote_expert_opt.cu` と `src/kernels/cuda/route_prior.cu`:
  ピア GPU / リモート expert のためのルーティング前処理。

---

## 3. リングバッファ（prefill の expert ストリーム）

prefill では top-10 × 8,192 トークン = 数万 expert 参照が発生する。全部を VRAM に置けないので、
**リング状のバッファに expert を流し込みながら GEMM する**。

```
   RAM                                              VRAM
   ┌──────────────┐    PCIe DMA (batched)   ┌──────────────────┐
   │ expert slot  │ ──────────────────────▶ │ ring[head]       │
   │ expert slot  │ ──────────────────────▶ │ ring[head+1]     │
   │ expert slot  │                         │ ...              │
   └──────────────┘                         └────────┬─────────┘
                                                     │ 消費
                                            ┌────────▼─────────┐
                                            │ moe_fused / MMQ  │
                                            └──────────────────┘
```

リングは 0.1.39b 以降 **バイト単位でサイズ指定**（#583）。環境変数 `STRATA_RING_BYTES` で調整し、
`--pcie-frac`（ブランチ f-142）で「PCIe 帯域のうち何割を expert に割くか」を決める。
`BATCHED_DMA.md` に、1 回の DMA を大きくまとめる方が速い理由が書かれている。

関連: `STRATA_PREFILL_CPU_SHARE`（CPU に expert を何割回すか）、`STRATA_ROUTE_TAIL_SKIP`。

---

## 4. KV キャッシュの 3 層構造

> 図: [05 KV キャッシュ 3 層 — hot 64K / コールド / 圧縮](../diagrams/kv-cache.svg)

- **ホット**: VRAM。直近 64K トークン程度。
- **コールド**: ホスト RAM。`kv_stream.cu` が PCIe リングで必要分だけ戻す。
  `KV_PREFETCH` で先読み。
- **圧縮**: `kv_q8.cu`（int8）と `kv_q4.cu`（4bit）。`cvec.cu` の `fwht256` は
  Hadamard 回転で外れ値を均してから量子化する（回転しないと量子化誤差が偏る）。

QSA（疎な注意）では、さらに **indexer** が「どのセルを見るか」を top-k で選ぶ。
選ばれたセルだけを `kv_gather_kernel` で集めるので、実効の読み出しは top-k 分で済む。

---

## 5. PLE（28.8GB の n-gram テーブル）

`--ple-io direct`（既定）では、テーブルを **mmap せず O_DIRECT 相当で直接読む**。
1 トークンあたり数行だけなので、ページキャッシュを汚さない方が得。
`src/kernels/cuda/ple.cu` の `history_advance_kernel` が履歴を進め、
`gnorm_kernel` → `gate_kernel` → `conv_kernel` の順で PLE の寄与を残差に足す。

---

## 6. SMEM の使い方（カーネル側の視点）

| 用途 | サイズ | 登場ファイル |
|---|---|---|
| iq4nl コードブック | 16 B | `s_gemv.cu`（全 GEMV） |
| QSA のスコア配列 | 最大 8 KB/head（n_ids ≤ 2051） | `qsa.cu` の `qsa_attend` |
| prefill の重みタイル | 数十 KB、二重バッファ | `moe_fused_iq.cu`, `moe_fused.cu` |
| コードブック全体（i-quant） | 数百 B〜数 KB | `moe_fused_iq.cu` |

> **16 バイトの話**が一番おいしい。`s_gemv.cu` は IQ4_NL のコードブックを
> `__constant__` から **shared に移して 2.12 倍**にした。理由は「constant メモリは
> **全レーンが同じアドレスを読む**ときに速い。非線形コードブックはレーンごとに違う index を読むので
> 最悪のパターン」。この測定（`attn_qkv` IQ4_XS 10240×2560 で 0.2032ms → 0.0960ms）は
> 第 3 章に全文とともに載せてある。

---

*次: [02 量子化フォーマット](02_quant.md)*

