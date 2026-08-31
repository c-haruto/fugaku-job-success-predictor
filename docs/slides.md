---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  :root {
    --ink: #262626;
    --muted: #737373;
    --accent: #0F766E;
    --line: #E5E5E5;
    --panel: #F7F7F5;
  }
  section {
    font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", "Segoe UI", sans-serif;
    color: var(--ink);
    background: #ffffff;
    padding: 64px 76px;
    border-top: 6px solid var(--accent);
  }
  section.lead {
    border-top: none;
    padding: 0;
    position: relative;
    overflow: hidden;
  }
  .title-grid {
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    position: absolute;
    inset: 0;
  }
  .title-text {
    padding: 64px 60px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .title-image {
    position: relative;
    overflow: hidden;
  }
  .title-image p {
    margin: 0;
    position: absolute;
    inset: 0;
  }
  .title-image img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  section.lead h1 {
    color: var(--ink);
    font-size: 2.1em;
    margin-bottom: 0.2em;
    padding-bottom: 0.35em;
    border-bottom: 5px solid var(--accent);
    display: inline-block;
  }
  section.lead p {
    color: var(--muted);
    font-size: 1.05em;
  }
  .eyebrow {
    color: var(--accent);
    font-weight: 700;
    letter-spacing: 0.08em;
    font-size: 0.8em;
    margin-bottom: 0.6em;
  }
  .github-badge {
    display: inline-block;
    margin-top: 1.4em;
    padding: 0.4em 0.9em;
    border: 1px solid var(--line);
    border-radius: 4px;
    font-size: 0.75em;
    color: var(--muted);
  }
  .github-badge a {
    color: var(--ink);
    text-decoration: none;
    font-weight: 600;
  }
  h2 {
    font-size: 1.5em;
    color: var(--ink);
    padding-left: 0.5em;
    border-left: 8px solid var(--accent);
    margin-bottom: 0.9em;
  }
  ul {
    line-height: 1.7;
  }
  li strong {
    color: var(--accent);
  }
  li::marker {
    color: var(--accent);
  }
  pre {
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 6px solid var(--accent);
    border-radius: 4px;
    font-size: 0.85em;
  }
  section::after {
    color: var(--muted);
  }
---

<!-- _class: lead -->

<div class="title-grid">
<div class="title-text">

<div class="eyebrow">FUGAKU JOB SUCCESS PREDICTOR</div>

# 富岳ジョブ成功率<br>予測Webアプリ

ジョブを投入する前に、成功確率をニューラルネットワークで予測する

<div class="github-badge">GitHub: <a href="https://github.com/c-haruto/fugaku-job-success-predictor">c-haruto/fugaku-job-success-predictor</a></div>

</div>
<div class="title-image">

![](assets/fugaku.jpg)

</div>
</div>

<!--
発表者メモ:
・自己紹介、発表時間の目安などをここに書く
-->

---

## 【2】目的

- スーパーコンピュータ「富岳」では、ジョブが**失敗して終了**すると、
  確保していた計算資源・時間がそのまま無駄になってしまう
- 「投入する前に、この設定で成功しそうかどうか分かれば」という発想からスタート
- ノード数・実行時間・メモリ上限などの**投入条件だけ**から、
  そのジョブが成功するかをその場で予測するツールを作った

---

## 【3】技術説明 ①: ニューラルネットワークの仕組み

**使用データ**: F-DATA（富岳の実ジョブ実行ログ、約37ヶ月分・約2500万件）を学習に使用。多層パーセプトロン（MLP）による二値分類（成功/失敗）

![w:1050](assets/nn_diagram.svg)

正解とのズレ（損失）を誤差逆伝播法で計算し、重みを少しずつ調整して学習する

---

## 【3】技術説明 ②: 学習の工夫とシステム構成

- **クラス不均衡対策**: 少数派の失敗クラスの損失を重み付けして学習
- **確率較正（Platt scaling）**: 検証データで確率を較正し、
  「成功確率○%」が実際の成功率に近づくよう補正
- **データリーク防止**: 実行後にしか分からない情報
  （消費電力・実行時間など）は不使用。ユーザーID・ジョブ実行環境も、
  実運用で意味を持たない/時期依存で不安定と判明し除外
- **構成**: ブラウザ(HTML/JS) → FastAPI → PyTorchモデル+較正器 → 確率を返す

---

## 【4】このアプリができること（デモ）

<!-- ここはデモに合わせて自由に編集してください -->

- ノード数・要求時間・メモリ上限・周波数・投入時刻を入力すると、
  その場で成功確率が表示される
- 条件を変えると確率がどう動くか、その場で比較できる

---

## 【5】感想

- 実際にR-CCSを訪れた経験を、このWebアプリの制作に活かせたことが楽しかった
- 直近の生の運用データも学習に使いたかったが、匿名化が期間内に間に合わず実現できなかった
- データの性質上の限界はあるものの、富岳の運用状況は急変することがあるため、今後はそれにインタラクティブに対応できるようなアプリへと発展させたいと思った
