# Strata CUDA ソース読解

[Strata](https://github.com/Niko1221/Strata)（Qwen3.8-Flash-Next 125B MoE 推論エンジン、CUDA/HIP 実装）の
**57 ファイル / 34,312 行**を読む本。1 枚の GPU で 125B MoE を回す仕組みを、行番号から追う。

- 公開サイト（GitHub Pages / 論文調レイアウト）: https://tjm8874.github.io/strata-cu-readthrough/
- 対象リビジョン: Strata **v0.1.49 / HEAD `61b3fb5`**
- 全 9 章（00–08）＋ 図 7 枚

## 章立て

| # | 章 | 内容 |
|---|---|---|
| 00 | 総覧 | 125B MoE を 1 枚で回す分割（VRAM / RAM / SSD）と数字の地図 |
| 01 | メモリ階層 | 4 層の置き場所、pinned リング、doorbell、帯域の梯子 |
| 02 | 量子化 | S2（Q2_0）の 20 バイト・ブロック、Q8 系、i-quant の配置 |
| 03 | Decode GEMV | `s_gemv.cu` 全行注釈。1 スレッド 1 出力行の内側 |
| 04 | MoE | router、3 つの行き先、combine、実測タイミング |
| 05 | 注意機構と GDN | qsa / indexer / GDN の再帰、KV の 3 層 |
| 06 | Prefill | 8,192 チャンク、融合カタログ、magic float、parity の証明 |
| 07 | その他 | Philox、MTP 検証、PLE の 3 経路、GR、runtime の毒 |
| 08 | 目録 | 全 57 ファイルの表、「1 ファイル 1 事件」 |

## リポジトリ構成

```
docs/*.md             原稿（このリポジトリの正）
diagrams/*.svg        図の原本（暗色版。ポータル用）
diagrams/png/         同上 PNG
diagrams/light/       白背景版 SVG / PNG（公開サイト用、2 倍解像度）
index.html            サイバー調ハブ（ローカルポータル用）
site/                 公開サイトのビルド出力（GitHub Pages はこのサブツリー）
tools/mkbook.mjs      md → サイバー調 HTML（ポータル用）
tools/build-paper.mjs md → 論文調 HTML（公開用、site/ を生成）
tools/paperfig.mjs    図を白背景へ再配色して PNG 化
tools/sync-portal.sh  ローカル Studio Hub へ同期
```

## ビルド

```sh
node tools/paperfig.mjs      # diagrams/light/ を生成（要 headless Chrome）
node tools/build-paper.mjs   # site/ を生成
```

`paperfig.mjs` は playwright のキャッシュにある `chrome-headless-shell` を使う（macOS 前提）。
`marked` はグローバルに無い場合、DSH 同梱のもの（`/Users/ts/DSH/node_modules/.pnpm/marked@18.0.11`）を指しているので、
別環境では `npm i marked` して import 先を変える。

## 注意

- 公式ドキュメントではない。推測で埋めた箇所は本文に「推測」と明記している。
- 行番号・定数は v0.1.49（`61b3fb5`）時点のもの。上流が動いたら `docs/08_catalog.md` の表と突き合わせて更新する。
- 本文・図の著作権は本リポジトリの作者に帰属。Strata 本体のライセンスは上流リポジトリに従う。
