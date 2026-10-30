# 08. ファイルカタログ — 57 本の .cu を 1 枚に

> `src/kernels/cuda` 45 本 + `src/prefill` 9 本 + `src/core` 3 本 = **57 ファイル / 34,312 行**。
> 各行数は `wc -l` 実測。章はこの読本のどこで詳述しているか。

---

## 0. 読む順序（おすすめ）

1. `dequant_s2.cu`（51 行）— 最小の「量子化を解く」カーネル。ここで **S2 のレイアウト**を掴む
2. `s_gemv.cu`（694 行）— decode の主役。[03](03_decode_gemv.md)
3. `router_top10.cu`（497 行）— 「演算量≠時間」の実例。[04](04_moe.md)
4. `qsa.cu`（821 行）— attention とキャプチャ監査。[05](05_attention_gdn.md)
5. `gdn.cu`（248 行）— 再帰がバリア不要な理由。[05](05_attention_gdn.md)
6. `moe_fused.cu`（728 行）— magic float。[06](06_prefill.md)
7. `sampler.cu`（1393 行）— 「順序が内容」。[07](07_misc.md)

---

## A. 量子化・活性・GEMV（decode の土台）

| ファイル | 行 | 役割 | 章 |
|---|---|---|---|
| `dequant_s2.cu` | 51 | S2（Q2_0）の最小デカント。1 スレッド = 64 要素ブロック、256 スレッド | [02](02_quant.md) [03](03_decode_gemv.md) |
| `s2_gemv.cu` | 63 | S2 GEMV の素朴版（1 スレッド/出力行、128 スレッド、ラッパで同期） | [03](03_decode_gemv.md) |
| `s2_gemv_quads.cu` | 93 | 4 要素/コードロード。L4 で 66 G weights/s を測った仮説検証 | [03](03_decode_gemv.md) |
| `s2_gemv_fast.cu` | 153 | 名前は fast だが**速くない**（両てこが中立〜悪化、分散 20% が効果を隠す） | [03](03_decode_gemv.md) |
| `s2_gemv_q8.cu` | 96 | Q8_0 活性を食う S2 GEMV（ggml と同じ契約） | [02](02_quant.md) [03](03_decode_gemv.md) |
| `s_gemv.cu` | 694 | **本命**。4 カーネル（`s_gemv_kernel` / `_split` / `_q8k` / `_q8_split`）+ SForm | [03](03_decode_gemv.md) |
| `quantize_act.cu` | 352 | 活性の量子化（Q8_0 / Q8_K）+ f32_to_f16 の罠（round 198） | [02](02_quant.md) |
| `dequant_bf16.cu` | 304 | BF16・Q8_0 のデカント（ggml-quants.c から転記） | [02](02_quant.md) |
| `iq_kernels.cu` | 3614 | IQ1_M〜IQ4_XS・Q4_K 系のドット積/デカント/q8_1 量子化。**最大ファイル** | [02](02_quant.md) [03](03_decode_gemv.md) |
| `native_mmvq.cu` | 2655 | llama.cpp MMVQ の移植（ncols 1..8、STRATA_Q8_PACKED、interleaved 列） | [02](02_quant.md) |
| `bf16_gemv.cu` | 141 | BF16 GEMV（`s_gemv` を使わない理由はヘッダ） | [02](02_quant.md) |
| `native_bf16.cu` | 383 | ピン留め CUDA の BF16/F32 MMVF（router・PLE が使う） | [04](04_moe.md) |
| `q8_1_il.cuh` | 99 | 2..8 列の q8_1 活性を interleaved に複製（multi-column ドット用） | [02](02_quant.md) |

---

## B. MoE（router・shared expert・grouped）

| ファイル | 行 | 役割 | 章 |
|---|---|---|---|
| `router_top10.cu` | 497 | top-10 routing。3 実装（CUDA / AMD fast / gfx906 wave） | [04](04_moe.md) |
| `native_router.cu` | 232 | ピン留め CUDA の route / route_multi / route_resident_k | [04](04_moe.md) |
| `shared_expert.cu` | 419 | 共有 expert（silu は gate 側、スカラ gate、**BF16 同士**の契約） | [04](04_moe.md) |
| `s2_expert_grouped.cu` | 1236 | grouped GPU expert。int8 ドット + チャンク float、活性 4×4 転置 | [04](04_moe.md) |
| `native_moe.cu` | 170 | combine / combine_k10_vec4 / S26 LFUSE | [04](04_moe.md) |
| `route_prior.cu` | 79 | `STRATA_ROUTE_PRIOR=lambda`: **キャッシュ条件付き routing**（opt-in、出力が変わる） | [04](04_moe.md) |


---

## C. attention・KV・位置エンコード

| ファイル | 行 | 役割 | 章 |
|---|---|---|---|
| `qsa.cu` | 821 | QSA キャッシュ/インデクサ/attention の入口 7 つ。キャプチャ監査 | [05](05_attention_gdn.md) |
| `qsa_select.cu` | 1365 | ブロックスコアと top-k 選択。世代順に warp / tc / wmma / GEMM / cluster | [05](05_attention_gdn.md) |
| `qsa_decode_attn.cu` | 758 | 8 要素ロード（f16/q8/q4）+ チャンク注意 | [05](05_attention_gdn.md) |
| `qsa_prompt_attn.cu` | 1525 | prefill 側注意（mma16816 int8 / i8 / 884） | [05](05_attention_gdn.md) [06](06_prefill.md) |
| `native_qsa.cu` | 167 | norm / norm_grouped / gate（llama.cpp から） | [05](05_attention_gdn.md) |
| `native_qsa_indexer.cu` | 308 | indexer キーの追加・pooled value・norm_scale | [05](05_attention_gdn.md) |
| `native_qsa_score.cu` | 213 | score（mma）と scalar_score | [05](05_attention_gdn.md) |
| `native_flash_attn.cu` | 212 | llama.cpp fattn-vec の特化（D=256, ncols=1, F16/F16） | [05](05_attention_gdn.md) |
| `kv_q8.cu` | 147 | KV の Q8 追記/収集 | [05](05_attention_gdn.md) |
| `kv_q4.cu` | 241 | KV の Q4_0 + Hadamard 回転（PR #21） | [05](05_attention_gdn.md) |
| `kv_stream.cu` | 286 | KV リング（copy / reset / ring、プレフィックス和、16 B/スレッド） | [01](01_memory.md) [05](05_attention_gdn.md) |
| `cvec.cu` | 174 | `fwht256`（Hadamard 回転）+ `cvec_apply` | [07](07_misc.md) |
| `rope.cu` | 136 | NEOX 部分 RoPE（旧経路） | [05](05_attention_gdn.md) |
| `native_rope.cu` | 245 | ピン留め ggml rope の数値契約（NEOX ペア、部分 64/256、mrope 表） | [05](05_attention_gdn.md) |

---

## D. GDN（ゲート付きデルタネット）

| ファイル | 行 | 役割 | 章 |
|---|---|---|---|
| `gdn.cu` | 248 | conv / l2 / beta-gate / out-norm / 再帰。**バリア不要**の理由 | [05](05_attention_gdn.md) |
| `fused_gdn.cu` | 166 | 融合版（1 カーネルで pre を畳む） | [05](05_attention_gdn.md) |
| `native_gdn.cu` | 122 | llama.cpp gated_delta_net.cu の移植（warp が 1 列を担当） | [05](05_attention_gdn.md) |
| `native_gdn_preprocess.cu` | 199 | conv_silu / l2_norm / beta_sigmoid / gate_softplus / out_norm | [05](05_attention_gdn.md) |
| `verify_kernels.cu` | 1357 | 投機検証ウィンドウ。**単一トークン版からの書き写し** | [07](07_misc.md) |

---

## E. GR（ゲート付き残差・ハイパーコネクション）

| ファイル | 行 | 役割 | 章 |
|---|---|---|---|
| `gr.cu` | 500 | `gr_read` / `gr_write`（分割版） | [07](07_misc.md) |
| `fused_gr.cu` | 1991 | 融合カーネル群。`gr_down_finish_kernel`、norm/up の遅延隠蔽 | [07](07_misc.md) |
| `native_gr_norm.cu` | 109 | `weighted_rms_norm`（連続 F32） | [07](07_misc.md) |
| `native_gr_postops.cu` | 161 | down_silu / pre_gated / post / post_multi | [07](07_misc.md) |

---

## F. PLE（28.8 GB の n-gram テーブル）

| ファイル | 行 | 役割 | 章 |
|---|---|---|---|
| `ple.cu` | 390 | PLE ブロックの GPU 半分。legacy は CPU ggml グラフ準拠 | [07](07_misc.md) |
| `native_ple_postops.cu` | 289 | gate / broadcast / conv_residual / rms_rep（+batch 版） | [07](07_misc.md) |

---

## G. 出力（サンプラ・糊）

| ファイル | 行 | 役割 | 章 |
|---|---|---|---|
| `sampler.cu` | 1393 | サンプラ連鎖。3 実装が同じトークン、Philox はカウンタベース | [07](07_misc.md) |
| `elementwise.cu` | 446 | 埋め込み gather・gdn_gate・**doorbell**・マップ済みコピー | [07](07_misc.md) |
| `s26_tsum.cuh` | 71 | 複数の warp 和を同時に（P 個の和、別々の xor バタフライとビット一致） | [04](04_moe.md) |


---

## H. prefill（`src/prefill`）

| ファイル | 行 | 役割 | 章 |
|---|---|---|---|
| `kernels.cu` | 2214 | prefill の注意・GDN・PLE・postops。融合カタログ（F-1/F-2/S23/C-3/Aurora/D-2） | [06](06_prefill.md) |
| `gemm.cu` | 873 | cuBLAS / hipBLASLt ラッパ。in-place FP16→FP32、DBG_NAN、Windows の古いエラー | [06](06_prefill.md) |
| `wmma_gemm.cu` | 737 | RDNA3 WMMA（gfx1100..1151）。`STRATA_WMMA_GFX11` はビルド定義 | [06](06_prefill.md) |
| `moe_fused.cu` | 728 | 自前 int8 テンソルコア MoE（magic float、m16n8k32 の k 順） | [06](06_prefill.md) |
| `moe_fused_iq.cu` | 1215 | i-quant を int8 に復号して同じ道へ | [06](06_prefill.md) |
| `moe_mmq.cu` | 298 | llama.cpp MMQ（ピン留め checkout から） | [06](06_prefill.md) |
| `ggml_cuda_host.cu` | 147 | MMQ が参照する ggml-cuda のホストシンボル | [06](06_prefill.md) |
| `gdn_rec_parity.cu` | 1383 | prefill 再帰 vs decode のビット一致検証 | [06](06_prefill.md) |
| `gemm_bf16_parity.cu` | 155 | BF16 GEMM 経路の一致検証（sm_80 未満は FP16 経由） | [06](06_prefill.md) |
| `wmma_gemm.h` | 47 | 上の宣言（HIP/CUDA 両対応） | [06](06_prefill.md) |

---

## I. ランタイム（`src/core`）

| ファイル | 行 | 役割 | 章 |
|---|---|---|---|
| `device.cu` | 364 | CUDA/HIP 抽象・poison（NaN）・アーキ照合 | [07](07_misc.md) |
| `pinned.cu` | 800 | ピンド arena と並列 expert ロード（64 ビット seek、ヒュージページ） | [01](01_memory.md) [07](07_misc.md) |
| `remote_expert_opt.cu` | 197 | ピア/リモート GPU の expert 経路 | [01](01_memory.md) |

---

## J. 「1 ファイル 1 事故」早見表

このコードベースの価値は、**事故とその修正理由がコメントに残っている**ことにある。
1 行で思い出せるようにまとめる。

| ファイル | 事故 |
|---|---|
| `s_gemv.cu` | barrier の前に early return（デッドロック） |
| `s2_gemv_fast.cu` | 「速い」と名付けたら速くなかった（分散 20% が効果を隠す） |
| `s2_expert_grouped.cu` | gate-major の書き方が >1 hit で別 expert の行を読む |
| `shared_expert.cu` | `const float*` で BF16 を読む → 4.5 GiB arena 内を静かにオーバーリード |
| `quantize_act.cu` | f32_to_f16 が**有限のオーバーフロー値を全部 NaN** に（fixture が O(1) だけだった） |
| `router_top10.cu` | O(n²) の rank-by-counting が moe_route の 63% |
| `qsa.cu` | `__fadd_rn`（単精度）を書いて pooling 全体が float に（0.4 ulp、許容誤差は全部通る） |
| `qsa.cu` | キャプチャで int64 引数が凍結 → 「文脈が無い」のに尤もらしい出力 |
| `sampler.cu` | penalties を 2 度掛け・min_p を top_p の前に（#53、分布でしか分からない） |
| `verify_kernels.cu` | 128 スレッド版で 7% の出力が 1 ulp 違う（#937 のウィンドウ数報告） |
| `ple.cu` | n-gram kernel の転置を取り違え（`ple_layer_xcheck.cpp` L79–83） |
| `elementwise.cu` | グラフキャプチャで `cudaEventRecord` が黙って消える → doorbell |
| `fused_gr.cu` | AMD (gfx906) 向け最適化を CUDA でもビルドする際のマクロ |
| `prefill/wmma_gemm.cu` | コンパイラマクロのガードが「return false」スタブを選び、WMMA が一度も走らない |
| `prefill/gemm.cu` | Windows の hipBLAS が成功時に古い `hipErrorInvalidValue` を残す（#247/#325） |
| `core/pinned.cu` | Windows の `fseek` が 32 ビットで 42.9 GB パックを壊す |

---

## K. 全体の数字

| 指標 | 値 |
|---|---|
| .cu / .cuh ファイル | 57（+ ヘッダ多数） |
| 行数 | 34,312（3 ディレクトリ合計） |
| 最大ファイル | `iq_kernels.cu` 3,614 行 |
| 入力形式 | 13 種 → GEMV では 3 コード幅 + 属性 |
| expert 総数 | 24,576（48 層 × 512） |
| top-k | 10 |
| パック全体 | 38 GiB（うち Q2_0 が 31.64 GiB） |
| PLE テーブル | 28.8 GB（SSD） |

---

→ 次は [図一覧](../index.html) の図と [index.html](../index.html) を参照。

