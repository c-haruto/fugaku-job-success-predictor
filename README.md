# 富岳ジョブ成功率予測Webアプリ

スーパーコンピュータ「富岳」にジョブを投入する前に入力する条件（ノード数、
要求経過時間、要求メモリ上限、要求周波数、投入時刻など）から、そのジョブが
成功する確率をニューラルネットワーク（多層パーセプトロン）で予測し、
ブラウザ上で確認できるプロトタイプアプリです。

データセットには [F-DATA](https://doi.org/10.5281/zenodo.11467483)
（Antici et al., *Scientific Data*, 2025）を使用しています。

> **注意**: 本アプリはプロトタイプ／教育目的のデモです。詳細は
> [`docs/training_report.md`](docs/training_report.md) を参照してください。

## すぐに試したい場合

**学習済みモデル（`src/models/`）とサンプルデータは本リポジトリに同梱済み**
なので、F-DATAのダウンロードや再学習をしなくても、下記コマンドだけで
Webアプリを起動できます。

```bash
pip install -e .
uvicorn src.main:app --reload
```

http://127.0.0.1:8000/ を開けばすぐに予測を試せます。

## データリークについて（最重要）

「ジョブを投入する前」に成功確率を予測するアプリである以上、モデルに
入力してよいのは **投入時点（実行開始前）で値が確定している情報だけ** です。
実行後にしか分からない情報（実測の消費電力・flops・実行時間など）を使うと、
一見精度が高く見えても実運用では役に立たないモデルになってしまいます。

本プロジェクトでは、この切り分けを [`src/config.py`](src/config.py) の
コメントと定数（`SAFE_RAW_COLUMNS` / `LEAKED_COLUMNS`）で明示しています。
詳しい設計方針は [`docs/feature_leakage.md`](docs/feature_leakage.md) に
まとめています。

- **使用した特徴量**: 要求ノード数、要求経過時間上限、要求ノード時間
  （ノード数×時間）、要求メモリ上限、要求周波数、投入時刻・曜日
- **使用しなかった特徴量**: 実測の消費電力・flops・メモリ帯域・演算強度・
  性能クラス・実際の実行時間・CPU時間・アイドル時間など、実行後にしか
  分からない全ての指標
- **あえて使わなかった特徴量**: ユーザーID（`usr`。F-DATAの匿名化IDは
  実際の富岳ユーザー名に対応しないため）、ジョブ実行環境（`jobenv_req`。
  ある時点を境に分布が恒久的に切り替わっており、未来の期間に汎化しない
  ため）。いずれも詳細は [`docs/feature_leakage.md`](docs/feature_leakage.md)

## ディレクトリ構成

```
.
├── README.md
├── AGENTS.md              # AIコーディングエージェント利用の記録
├── pyproject.toml         # 依存ライブラリ
├── assets/
│   ├── images/logo.png    # アプリのロゴ
│   └── icons/favicon.png  # ファビコン
├── docs/
│   ├── dataset.md              # F-DATAの説明・EDA結果
│   ├── feature_leakage.md      # 安全な特徴量/リーク特徴量の切り分け詳細
│   ├── feature_list_full.csv   # F-DATA原本の全45特徴量一覧
│   ├── training_report.md      # 学習・評価結果レポート（自動生成）
│   └── figures/                 # 混同行列・ROC曲線・学習曲線（自動生成）
├── data/
│   ├── raw/                # ダウンロードしたparquet（サイズが大きいため.gitignore対象）
│   └── sample.csv          # 前処理後の特徴量サンプル（自動生成）
└── src/
    ├── main.py             # FastAPIアプリ本体（エントリポイント）
    ├── config.py           # 特徴量の定義・データリーク防止のための設定
    ├── data_prep.py        # 特徴量エンジニアリング
    ├── download_data.py    # F-DATAのparquetファイルを並列ダウンロードするスクリプト
    ├── train.py            # モデルの学習・評価・確率較正スクリプト
    ├── model.py            # PyTorchのMLP定義
    ├── inference.py        # 学習済みモデルを読み込んで推論するモジュール
    ├── models/             # 学習済みモデル・前処理器・較正器など（自動生成）
    └── static/             # シンプルなHTMLフロントエンド
        ├── index.html
        ├── style.css
        └── app.js
```

## モデルについて

- 多層パーセプトロン（全結合層 → ReLU → Dropout を2段、出力1ユニット）
  による二値分類（PyTorch実装、[`src/model.py`](src/model.py)）
- クラス不均衡（失敗ジョブは少数派）に対応するため、`sklearn`の
  `balanced`と同じ考え方でクラスごとに損失の重みを変えている
- クラス不均衡対策の重み付き損失は、出力される確率の値そのものを歪める
  副作用があるため、検証(val)セットで確率較正（Platt scaling）を行い、
  「成功確率〇%」という表示が実際の成功率に近くなるよう補正している
  （詳細は [`docs/training_report.md`](docs/training_report.md)）
- 評価指標: Accuracy、Precision/Recall/F1（成功・失敗の両クラス）、
  ROC-AUC、Brierスコア。詳細は [`docs/training_report.md`](docs/training_report.md)

## モデルの限界

本アプリが利用できるのは、ジョブ投入時にユーザー自身が指定する条件
（ノード数・時間・メモリ上限・周波数・投入時刻）だけであり、ユーザーの
識別情報や、実行後にしか分からない情報は一切使っていない（詳細は
[`docs/feature_leakage.md`](docs/feature_leakage.md)）。そのため、実際の
ジョブ失敗の主要因がコードのバグや設定ミスなど、ジョブ投入条件だけからは
読み取れない要素に依存している場合、予測性能には限界がある。実際の評価
指標（ROC-AUCや失敗クラスのRecallなど）は
[`docs/training_report.md`](docs/training_report.md) を参照。

### 成功率が下がりやすい条件・上がりやすい条件（参考）

実際にモデルへ様々な条件を入力して確認したところ、次のような傾向が
見られた（あくまで学習データ上の統計的な傾向であり、因果関係を示す
ものではない）。

- **メモリ上限を指定しない**ジョブは、指定するジョブより成功率が
  明確に低く出る（同条件で指定ありが95%超に対し、指定なしは75〜85%
  程度になることが多い）。データ上、メモリ上限を明示するジョブの方が
  作り込まれた安定運用のジョブである傾向が強いためと考えられる。
- **要求周波数**は `2000` MHz が他の値（`1600` / `2200` MHz）より
  成功率が低く出やすい。
- ノード数・経過時間・投入時刻・曜日の影響はあるが、上記2つほど大きくは
  なく、組み合わせ方によって変動する（単純に「大きい・長いほど危険」
  という単調な関係ではない）。


## ライセンス・出典

- データセット: F-DATA (Antici, F., Bartolini, A., Domke, J., Kiziltan, Z.,
  Yamamoto, K., *Scientific Data*, 2025)、
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) のもとで
  Zenodoに公開されている（DOI:
  [10.5281/zenodo.11467483](https://doi.org/10.5281/zenodo.11467483)）。
  本アプリはこのデータセットを用いて学習しており、`data/sample.csv` に
  前処理後の特徴量の一部（500件、匿名化・数値変換済みで元データの
  識別情報は含まない）を再配布している。
- 参考リポジトリ: https://github.com/francescoantici/F-DATA
