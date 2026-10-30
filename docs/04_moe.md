# 04. MoE — 512 個から 10 個を選び、GPU と CPU で分担する

> 対象: `src/kernels/cuda/router_top10.cu`（497）、`s2_expert_grouped.cu`（1236）、
> `shared_expert.cu`（419）、`native_moe.cu`（170）、`native_router.cu`（232）
>
> 図: [04 expert の流れ — router → 3 つの行き先 → combine](../diagrams/expert-flow.svg) ／ [01 ストリームと doorbell — GPU と CPU を重ねる](../diagrams/streams-doorbell.svg)

## 0. 形の確認

Qwen3.8-Flash-Next 125B は **48 層 × 512 expert = 24,576 expert**。1 トークン・1 層あたり
**top-10** を選ぶ。1 トークンの decode で動く expert は 480 個、重みにして数十 GB。
だからこそ「全部 RAM に置き、VRAM にはホットな約 700 expert/GiB だけ」という設計になる（[01](01_memory.md)）。

1 層の MoE ブロックは 3 段階に分かれている（`include/strata/core/layer.hpp` の分割グラフ）:

```
block_layer_pre[l]  ──  attn → mixer → gr_write → ffn gr_read → moe_route
                                                        │
                                                        │  ring the doorbell（共有メモリの序列番号）
                                                        ▼
                        ┌────────────── GPU ──────────────┐   ┌──── host（CPU pool）────┐
                        │ shared expert（既定 shared-early） │   │ VRAM に居ない expert を    │
                        │ VRAM resident expert の計算        │   │ AVX-512 で計算（並行）     │
                        └──────────────┬──────────────────┘   └───────────┬────────────┘
                                       ▼                                  ▼
block_layer_post[l] ── moe_finish（重み付き合成 + shared 加算）→ gr_write
```

**doorbell** がこのエンジンの心臓部。graph capture 中は `cudaEventRecord` が黙って落ちるので（round 199）、
カーネルが「マップ済みピンドメモリにシーケンス番号を書く」ことでホストへ合図する。

---

## 1. Router — 289ms のトークンの 56% を食っていた場所

### 1.1 仕様は `ref/moe.py::router` の写し

```python
p   = softmax(logits)              # 全 expert 上で。選ばれた 10 個の上ではない
ids = stable argsort(-p)[:k]       # 同値は index 昇順
w   = gather(p, ids)
s   = max(sum(w), 2**-14)          # ggml_clamp(..., 2**-14, INF)
return ids, w / s
```

ソースのコメントが「**2 段構造は見た目の問題ではない**」と強調している:
選ばれた部分集合の上で softmax を取っても数学的には同じ（softmax はシフト不変）だが、
**再正規化の位置が動き、CLAMP は再正規化の一部**なので、gather 形式を再現しないと clamp が別の場所に落ちる。

### 1.2 「演算量が少ないから高速」という誤り

初版は **1 スレッド = 1 トークン**で、k 回のパスごとに全 expert の `exp` を再計算していた。
コメントの正当化はこうだった:

> 「トークンあたり 5,120 演算。expert matvec の 2.36e9 重みに比べて 3 桁小さいので、
> 最も単純な正しい版が十分速い」

> **それは演算量についての主張で、時間については何も言っていない。5,120 演算がすべて 1 スレッドの上に
> あったのに対し、matvec は数千スレッドで走っていた。**

実測（round 218、1 ブロック内を段階計測）:

```
  moe_layer    4.077 ms
  router_top10 3.387 ms      ← 2.6 MB しか読まないカーネル
  bf16 gemv    0.270 ms
```

**3.39 ms × 48 層 = 163 ms。289 ms のトークンの 56%。**
しかも `router_top10_parity` は 1 回呼んで id を確認するだけなので、これを見逃していた。

### 1.3 何を並列化し、何を意図的に直列に残したか

| 部分 | 判断 | 理由 |
|---|---|---|
| max | ツリー `fmaxf` | 厳密・順序非依存 → 直列走査とビット一致 |
| 512 個の `exp` | **並列、1 回だけ** | 旧版は 11 回計算していた。3.4 ms の正体はここ |
| 512 個の double 加算 | **1 スレッド、昇順のまま** | 約 1 µs。順序を変えると最下位ビットが変わる。コストは exp であってここではない |
| 選択 | **k パス**（後述） | O(n²) → 51 倍の仕事削減 |

```cpp
// the sum, ascending, on one thread: this is NOT parallelised
if (tid == 0) {
    double sum = 0.0;
    for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
    s_sum = sum;
}
```

**「無料の高速化は、値の静かな変更でもある」** — これが並列化しない理由として明記されている。

### 1.4 選択アルゴリズムの 2 段進化

**第 1 版: rank by counting**

```cpp
rank(e) = #{ f : p[f] > p[e] } + #{ f < e : p[f] == p[e] }
```

「安定降順ソート、同値は index 昇順」と等価で、k 回の直列 max を 1 パスにする。だが
**O(n²) = 512×512 = 262,144 回の shared ロードと比較**、実測 34.35 µs で `moe_route` の 63%。

> 「コメントは既に FP64 演算をループの外へ出したと記録している（262,144 回の double 乗算 → 512 回）。
> **だが O(n²) という“構造”は残っていた。コストなのは構造のほうだ。**」

**第 2 版: k パス（現行）**

```cpp
for (int i = 0; i < k; ++i) {
    float bv = -INFINITY;
    int bi = n_expert;                 // すべての実 index に負ける番兵
    for (int e = tid; e < n_expert; e += nt) {
        if (s_taken[e]) continue;
        const float pe = s_p[e];
        if (pe > bv) { bv = pe; bi = e; }    // 厳密な > が同値時に最小 index を残す
    }
    ... // ワープ/ブロック縮約、これも (ov > bv || (ov == bv && oi < bi))
    if (tid == 0 && ix < n_expert) { ids[...] = ix; weights[...] = v; s_taken[ix] = 1; }
}
```

`k × 512 = 5,120 比較`、**51 倍少ない**、そして**同じ順序**。
正しさの議論は「`e` を昇順に走査し厳密な `>` を使えば、同値のとき最小 index が残る。
これは rank ループが `else if (f < e && pf == pe) ++rank` と書いていた規則そのもの」。

さらに `p[e]` の **262,144 回の double 乗算**も 512 回に巻き上げられた:

```cpp
float* s_p = (float*) (s_ex + n_expert);
for (int e = tid; e < n_expert; e += nt) s_p[e] = (float) (s_ex[e] * inv);
```

「**ビット一致であって、単に等価なのではない**: `p[e]` は同じ式を 513 回ではなく 1 回計算しただけ。」

### 1.5 AMD 版の話（設計の参考になる）

`router_top10_fast_kernel` は RDNA4 向けで、**直列部分を消してビット一致を保つ**技巧が詰まっている:

- 昇順 double 和の唯一の用途は `inv = (float)(1.0/sum)`。全項が正で最大値は厳密に 1 なので、
  直列和と任意の順序の和は 2⁻⁴³ 相対以内。**1.0/x と float キャストは単調**だから、
  区間 `[tree(1-2⁻⁴⁰), tree(1+2⁻⁴⁰)]` が 1 つの float に写るなら、それは直列和の float と同一。
  写らないとき（パリティテストで 1000 行に 1 行程度）だけ thread 0 が直列和を走らせる。
- 選択は wave 0 だけで、確率をレジスタに保持（1 レーン 16 個）。「最大 p かつ同値なら最小 index」は
  一意なので butterfly でも同じ expert が見つかる。

> **一般則**: 浮動小数の並列化は「一致しない」ではなく「**一致することを証明できるか**」で判断する。
> 証明できない部分だけ直列に残す。


---

## 2. Expert の計算 — 整数で dot して、float は 32 要素に 1 回

### 2.1 S2 の代数変形がすべてを決める

expert の重みは S2（Q2_0）。復号は `w = (code + bias) * d_w`。素直に書けば要素ごとに
「整数 → float 変換 → 乗算」だが、`s2_expert_grouped.cu` はまず**式を変形する**:

```
sum_j (c_j - 1) * d_w * x̂_j * d_x   ==   d_w * d_x * ( sum_j c_j·x̂_j  -  sum_j x̂_j )
```

> 両方の和は **INT8 × INT8** なので、どちらも `__dp4a`（1 命令 4 積和）。
> 整数で厳密、ループの中に非量子化が一切ない。

そしてスケールの粒度が違う: 重み `d_w` は **fp16、64 要素に 1 個**、
活性 `d_x` は **fp16、32 要素に 1 個**。だから float の仕事は
「**32 要素のチャンクあたり 1 回の乗算 + 1 回の加算**」になる。要素あたりではない。

> **ソースの自己申告**: 「和の順序は CPU のものではなく、**そうする事はできない**。
> レーン L はチャンク L, L+32, … を取り、部分和は shuffle で縮約される。CPU は全チャンクを
> 順番に自分のアキュムレータ形状で歩く。両者は float の丸めで一致し、ビットでは一致しない —
> `s_gemv_parity` が同じ理由で背負っているのと同じ契約。`bench/micro/moe_hit_parity.cu` が検査する。」

### 2.2 「チャンク内でどのコードがどの活性と組になるかは自由」

> `s = sum_e code_e * x_e` と `hx = sum_e x_e` は**厳密な整数和**（|s| ≤ 32·3·128）なので、
> 32 個の積をどう 4 個ずつのワードにまとめても、同じ 2 つの整数になる。
> ここで使うまとめ方はコードの転置を**コスト 0** で済ませる:
> `(w >> 2f) & 0x03030303` はコードワードの 4 バイトそれぞれのフィールド f
> （＝要素 `4b + f`、2 ワード目なら +16）を保持する。
> つまり**組み替えるのは活性のほう**で、チャンクごとに 1 回:

```
X[4h + f] = { x[16h + f], x[16h + 4 + f], x[16h + 8 + f], x[16h + 12 + f] }
```

> これはチャンクの各半分の 4×4 バイト転置で、**8 個の `__byte_perm`**。
> 旧カーネルはコード**バイト**あたり 7 個の整数演算と、行ごとに `hx` 用の 2 本目の dp4a チェーンを
> 使っていた。ここではコードワードが 16 要素あたり 7 個で、`hx` はチャンクあたり 1 回だけ和を取る。

**結果はビット一致**（`s`、`hx`、`dw * dx * (float)(s - hx)` は不変）。
「速くする = 値が変わる」ではないことの見本。

### 2.3 アラインメントからロードを減らす

> **32 回のバイトロードではなく、9 回の整列ワードロード。**
> 呼び出し側が活性行の 4 バイト整列を保証するので、チャンク c の int8 は `34c + 2` から始まる
> ＝整列ワードの 0 バイト目（c 奇数）か 2 バイト目（c 偶数）。ワードは整列して読み、
> funnel shift で位置を合わせる。9 個目のワードは 2 バイト目の場合だけ読み、しかもそれは
> 同じ行のブロック c+1 の内側にある — 両方の行長（80 と 20 チャンク）が偶数なので、
> **偶数のチャンクが行の最後になることはない**。

「行長が偶数」という一見どうでもいい性質が、端数処理を消すために使われている。

### 2.4 gate-major というバグ修正

> 最初の版は `gate_up[h*2FF + {0..FF-1}] = gate`、`[h*2FF + FF..] = up` と書いていた。
> どちらも**連続した範囲**を歩くので、hit が 2 個以上あると **hit 0 の UP 行を、他の hit の
> gate 行があるべき場所で読む**。gate-major にすると、2 つの他のカーネルにストライドを足すのではなく
> **食い違い自体が消える**。

> **教訓**: レイアウトの不一致は「ストライドを足して合わせる」より「レイアウトを変えて消す」。

### 2.5 bias の引き算を動かすと値が変わる

```cpp
// The CPU subtracts the weight bias after its eight FMA accumulators have been reduced. Moving the
// subtraction into each integer dot, as the legacy kernel does, changes rounding even with equal scales.
__global__ void activation_correction_kernel(const uint8_t* q8, const float* scales, float* hx, int chunks)
```

「スケールが等しいから同じ」は成り立たない。**どこで減算するかが丸めを決める**。
だから CPU 順序を再現する `cpu_order_projection_kernel` / `cpu_order_swiglu_kernel` /
`cpu_order_quantize_kernel` が別に存在し、パリティテストが両方を 1 プロセスで走らせる
（`STRATA_OLD_GROUPED=1` で A/B）。

---

## 3. グループ化 — 1 回の読みで複数のトークン

verify では 1 回のパスで最大 3〜5 トークンを検査する。同じ expert に複数トークンが行くなら、
**行を 1 回読んで全部のトークンに使う**のが正しい。

> 「Plan v0.3 P6: experts GROUPED — グループごとに 1 つの blob ポインタ（VRAM スロット、
> または PCIe 越しに読むマップ済みホスト blob）。グループの全エントリ（その expert に
> ルーティングされたトークン）を、**各行 1 回の読み**から計算する。」

```cpp
// Gate/up: 1 ブロック = 1 グループの GU_ROWS 行。グループの活性（各エントリのトークン行 x_q8_0 と
// その fp32 スケール）を shared に 1 回だけ整列ワードとして置き、各ワープが自分の行を歩き、
// レーンのコードチャンクを 1 回読んで全エントリと dot する。
__global__ void __launch_bounds__(256) gu_grouped_kernel(const unsigned long long* grp_ptr, ...)
```

### 3.1 8-way バンクコンフリクトの発見と修正

> 「上のレイアウトは **8-way バンクコンフリクト**。`xs_q[k][c*8+j]` はレーン L のワード j を
> `8L+j` に置くので、レーン `L, L+4, L+8, …` がチャンクの 8 回の読みすべてで同じバンクを共有する。
> さらに `chunk_dot` は（エントリだけで決まる）`hx` を行ごとに再計算していた。」

修正は「ワード優先」レイアウト:

```
xs_w[j][k * NC + c]    ワード j（再グループ化したチャンク）、レーンは連続ワードを読む
xs_dh[k * NC + c]      (dx, hx) — チャンクあたり float 1 個 + dp4a 8 回ではなく 64bit 1 回
```

「ワード優先だと書き込みも連続になる（フラット index `i` がそのまま `k * NC + c`）ので、
どちら側もコンフリクトせず、パディングも要らない。」さらに**各ワープが行を 2 本ずつ**扱う
（隣接行: gate `r` と up `r`）ので、shared 読みが両方に効き、**コードの展開が
エントリごとではなく行ごとに 1 回**で済む。

### 3.2 常駐判定もデバイスで行う

> 「Plan v0.3 P4: この層のルーティング先 expert のうちどれが常駐かは**デバイス上で**決める
> （静的な常駐行から）。リングと hit カーネルの間にホストのステップを挟まない。」

`hit_select_kernel`（1 ワープ、k ≤ 32）、`hit_select_multi_kernel`（最大 128 エントリ、4 ワープ、
ballot をエントリ順に詰める）、`add_hits_kernel`（部分和の加算）。
MTP 層のように全 expert が `base + id * blob` に常駐する場合は `group_resident_kernel` が
グループをデバイス上で組む。


---

## 4. shared expert — 7 つのカーネルのうち 6 つは説明がつく。1 つはつかない

### 4.1 仕様

```
h = silu(x @ gate_shexp.T) * (x @ up_shexp.T)     ← SILU は GATE にかかる。up ではない
h = h @ down_shexp.T
g = sigmoid(x @ gate_inp_shexp)                   ← トークンあたり【1 スカラー】
return h * g[:, None]
```

> ここには**何ラウンドも間違って信じられていた** 2 点がある。
> **SILU は gate テンソルにかかる**。逆の読みは自然で、形も正しく、そして間違い。
> そして `ffn_gate_inp_shexp` は `(n_embd,)` で出力は「トークンあたり 1 値」—
> **そのベクトルと隠れ状態の dot で得られるスカラー gate** であって、expert ごとの gate でも
> 次元ごとの要素積でもない。
>
> 結果は routed 出力に**加算**される。router で重み付けされるのでも、再正規化されるのでもない。

### 4.2 1 スレッドがモデル最大の単一カーネルだった

旧コメント:「n_embd は 2560 なので短い直列ループだし、並列性は他所にある。」
ソースの反論:

> **2560 回は短くない。しかも各回が DOUBLE の積和だから。このカードの double レートは float の
> 1/64。つまりこのループは、1/64 レートのユニットを通る 2560 段の依存チェーン**であり、
> 2560 回の非合体グローバルロードを 1 スレッドが発行することを数える前から、数十 µs。

実測:

```
  shared_expert  0.2374 ms / 0.8883 ms ブロック  = 26.7%
                 トークンあたり 11.4 ms
                 3 つの射影が読む量 5.5 MB、そのメモリ床 0.009 ms
```

> 「7 つのカーネルがあり、6 つは 640〜2560 要素の GEMV か要素演算。
> **帯域では説明できないのはこれ 1 つだけ。**」

修正は**チェーンを切る**だけで、精度は落とさない:

> 「double のアキュムレータは**残す**。`ref/gr.py` は float64 で縮約するし、`gr.cu` の
> `block_sum` も同じ理由で存在する — ツリーの和の順序は参照のものではないから、
> 順序が問題にならないだけの精度で守る。変えるのは**チェーン**で、1 スレッド 2560 項ではなく
> 10 項 + ツリー縮約。sigmoid を通した後では、dot の 1e-16 相対差は差ではない。」

### 4.3 5120 バイトの越境読み — 「もっともらしい嘘」

> **両オペランドは BF16。これは参照の契約であって、保存形式の話ではない。**
> `gate_inp_shexp` は BF16 重みなので、ggml は活性を `vec_dot_type`（＝BF16）に変換する。
> このカーネルはかつて `const float* w` を受けていたが、**ローダは既にテンソルを
> 2 B/要素に丸め直していた**。つまり 5120 バイトのバッファから 2560 個の float を読んだ —
> **テンソルの終端を 5120 バイト越えて**、4.5 GiB のアリーナの内側、誰もフォールトしない場所で。
> dot は天文学的に大きくなり、sigmoid は 1.0 に飽和し、`scale_kernel` が shared expert の
> 出力をそれで掛けた。
>
> **ここでの間違ったスカラーは静かな失敗モード**: sigmoid は被害を [0,1] に閉じ込めるので、
> 0.5 であるべき gate が 1.0 を読むと shared expert が 2 倍になり、**完全に有限で、
> 完全にもっともらしい logits** が出る。

「有界な関数」は GPU の範囲外読みと最悪の相性である。

### 4.4 f32 → f16 の NaN バグ（round 198）

> `f32_to_f16` はここに私設コピーとして存在した。round 198 がそれを見つけた —
> そして `quantize_act.cu` の同一コピーも — **有限値が fp16 をオーバーフローするとき
> 必ず NaN を返していた**（1e30, 65536, 1e45）。飽和して inf になるべきところで。
> 理由は `exp >= 31` が **f32 の inf/NaN と範囲外の有限指数を混同**していたから。
> このファイルのフィクスチャは全部 O(1) なので、見えなかった。

**テストが通る範囲が、テストの守る範囲**である。

---

## 5. 合成（combine）

```cpp
// GATED (S26 STRATA_LFUSE): `shared` は shared expert の未スケール出力、sg はトークンごとの生 gate logit。
// 行は shared_expert_multi と同様にここでスケールされ（native_scalar_sigmoid_multi_kernel の式、
// 続いて 1 回の丸め乗算 = scale_rows_kernel の `out *= g`）、その後これまで通り加算される。
__global__ void combine(const float* parts, const float* weights, ...);
__global__ void combine_k10_vec4(const float4* parts4, const float* weights, ...);
```

10 個の部分和に router の重みを掛けて足す。`combine_k10_vec4` は float4 で 4 要素ずつ。
S26（`STRATA_LFUSE`）では shared expert のスケールを合成側に融合し、カーネル 1 本を消している。

---

## 6. この章のまとめ — MoE が教える 5 つのこと

1. **演算量は時間ではない**。1 スレッドの 5,120 演算がトークンの 56% を食った。
2. **O(n²) の“構造”を疑う**。中の FP64 を外しても構造が残れば遅いまま。
3. **整数で dot し、float はチャンク単位で**。代数変形（bias を外へ）が dp4a を可能にする。
4. **並列化の可否は「ビット一致を証明できるか」で決める**。証明できない部分だけ直列に残す。
5. **有界な関数はバグを隠す**。sigmoid も fp16 飽和も、症状をもっともらしくする。

---

*次: [05 注意機構と GDN](05_attention_gdn.md) — QSA の indexer、KV の 3 階層、DeltaNet の状態。*

