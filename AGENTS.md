# AGENTS.md

このファイルは、本プロジェクトの実装に AI コーディングエージェント（Claude Code）を利用したことを記録するものです。

## 利用したエージェント

- **Claude Code**（Anthropic, Sonnet 5）

## エージェントに依頼した作業範囲

- F-DATA データセット（Zenodo, DOI: 10.5281/zenodo.11467483）のドキュメント調査（特徴量一覧の確認）
- データのダウンロードと前処理スクリプトの作成
- データリーク防止のための「投入時点で既知の特徴量」と「実行後にのみ判明する特徴量」の切り分け設計
- PyTorch による多層パーセプトロン（MLP）の学習・評価コード作成、確率較正（Platt scaling）の実装
- FastAPI バックエンド（推論API）とシンプルなHTMLフロントエンドの実装
- README / ドキュメント作成

## 人間によるレビュー・確認事項

- 特徴量の安全性（投入前情報のみを使っているか）は `src/config.py` の `SAFE_RAW_COLUMNS` / `LEAKED_COLUMNS` 一覧と `docs/feature_leakage.md` を人間が確認してください。
- 実データでの学習結果（`docs/training_report.md` 記載の Accuracy / Precision / Recall / F1 / ROC-AUC、混同行列）は F-DATA のほぼ全期間（2021年4月〜2024年4月、37ファイル）を用いた検証結果ですが、それでも予測性能には限界があります。詳細は `docs/training_report.md` と `docs/feature_leakage.md` を参照してください。
- 本アプリはあくまでプロトタイプ／教育目的のデモであり、実際の冨岳ジョブ運用における意思決定に使うことは想定していません。

## 再現方法

`README.md` の「セットアップ」「学習」「Webアプリ起動」の手順に従うことで、同様のエージェントを使わずに全工程を再現できます。
