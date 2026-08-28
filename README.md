# 冨岳ジョブ成功率予測Webアプリ

スーパーコンピュータ「冨岳」にジョブを投入する前に入力する条件（ノード数、
要求経過時間、要求メモリ上限、要求周波数、投入時刻など）から、そのジョブが
成功する確率をニューラルネットワーク（多層パーセプトロン）で予測し、
ブラウザ上で確認できるプロトタイプアプリです。

データセットには [F-DATA](https://doi.org/10.5281/zenodo.11467483)
（Antici et al., *Scientific Data*, 2025）を使用しています。

> **注意**: 本アプリはプロトタイプ／教育目的のデモです。実際の冨岳運用に
> おける意思決定への使用は想定していません。詳細は
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

「1. データのダウンロード」「2. モデルの学習」は、一から学習を再現したい
場合にのみ必要な手順です。

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
  実際の冨岳ユーザー名に対応しないため）、ジョブ実行環境（`jobenv_req`。
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

## セットアップ

Python 3.10以上を想定しています。

```bash
pip install -e .
```

（`pyproject.toml` の代わりに `pip install fastapi "uvicorn[standard]" torch pandas numpy scikit-learn pyarrow matplotlib joblib pydantic` を直接実行してもかまいません。）

## 1. データのダウンロード（再学習する場合のみ・任意）

同梱の学習済みモデルをそのまま使う場合はこの手順は不要です
（「すぐに試したい場合」を参照）。一から学習を再現したい場合のみ
以下を行ってください。

F-DATAは全38ファイル・約28GBあり、本プロジェクトでは`src/config.py`の
`TRAIN_FILES`（2021年4月〜2023年12月、33ファイル）・`VAL_FILES`（2024年
1-2月）・`TEST_FILES`（2024年3-4月）として、2021年3月（運用開始直後で
投入件数が極端に少ない日を含む月）を除くほぼ全期間を使う設定にしている。
以下のコマンドでこれらを`data/raw/`にダウンロードする。

```bash
python -m src.download_data
```

Zenodo側の帯域制限が1接続あたりに掛かっているため、複数ファイルを並列
ダウンロードする実装になっている（デフォルト16並列。`--workers`オプションで
変更可能）。それでも全量（約26GB）のダウンロードには回線速度次第で
数時間かかることがある。

より少ないデータで手早く試したい場合は、`src/config.py` の `TRAIN_FILES` /
`VAL_FILES` / `TEST_FILES` を書き換えてから実行すればよい（利用可能な
ファイル名は [F-DATAのZenodoページ](https://doi.org/10.5281/zenodo.11467483)
を参照）。

## 2. モデルの学習（再学習する場合のみ・任意）

```bash
python -m src.train
```

以下が自動生成されます。

- `src/models/model.pt` : 学習済みPyTorchモデル
- `src/models/preprocessor.joblib` : 特徴量の標準化・one-hotエンコーダ
- `src/models/calibrator.joblib` : 確率較正器（Platt scaling、下記参照）
- `src/models/categorical_options.joblib` : UIのドロップダウン用の選択肢
- `src/models/metrics.json` : 評価指標
- `docs/training_report.md` : 学習・評価結果のレポート（Markdown）
- `docs/figures/*.png` : 混同行列・ROC曲線・学習曲線
- `data/sample.csv` : 前処理後の特徴量のサンプル

約2500万行のデータを読み込むため、学習全体（データ読み込み〜評価〜保存）
で数十分程度かかります（GPUがあれば学習部分は高速化されます）。

## 3. Webアプリの起動

```bash
uvicorn src.main:app --reload
```

ブラウザで http://127.0.0.1:8000/ を開くと、フォームが表示されます。
ノード数・要求経過時間・メモリ上限・周波数などを入力して「成功確率を
予測する」を押すと、`POST /predict` APIが呼び出され、成功確率と簡単な
コメントが表示されます。

### API仕様（概要）

- `GET /` : フロントエンド（HTML）
- `GET /api/options` : 周波数の選択肢（学習データから取得）
- `POST /predict` : 予測API

`POST /predict` のリクエスト例:

```json
{
  "n_nodes": 4,
  "elapse_hours": 3,
  "freq_req": "2000",
  "submit_hour": 10,
  "submit_dow": 2,
  "mem_limit_gib": null
}
```

レスポンス例:

```json
{
  "success_probability": 0.839,
  "failure_probability": 0.161,
  "label": "success",
  "message": "比較的成功しやすい条件ですが、念のため設定を見直すと安心です。"
}
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

## ライセンス・出典

- データセット: F-DATA (Antici et al., *Scientific Data*, 2025),
  Zenodo DOI: [10.5281/zenodo.11467483](https://doi.org/10.5281/zenodo.11467483)
- 参考リポジトリ: https://github.com/francescoantici/F-DATA
