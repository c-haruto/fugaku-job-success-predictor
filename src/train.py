"""
モデルの学習・評価スクリプト。

実行方法:
    python -m src.train

処理の流れ:
    1. data/raw/ の parquet ファイル（config.TRAIN_FILES / VAL_FILES / TEST_FILES）を読み込む
    2. 投入時点で分かる特徴量のみを使って前処理する（config.py, data_prep.py 参照）
    3. 時系列に沿って train / val / test に分割する
    4. PyTorch の MLP を学習する（クラス不均衡を考慮した重み付き損失）
    5. test セットで Accuracy / Precision / Recall / F1 / ROC-AUC / 混同行列 を算出
    6. 学習済みモデル・前処理器・確率較正器・評価結果・図を保存する
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import config, data_prep
from .model import JobSuccessMLP


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
        ]
    )


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def train_one_run() -> dict:
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = config.DOCS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] parquetファイルを読み込み中...")
    raw = data_prep.load_raw_parquets(config.TRAIN_FILES + config.VAL_FILES + config.TEST_FILES)
    print(f"  読み込んだ行数: {len(raw):,}")

    print("[2/6] 特徴量エンジニアリング中（安全な特徴量のみ使用）...")
    features = data_prep.engineer_features(raw)

    print("[3/6] 時系列でtrain/val/testに分割中...")
    train_df, val_df, test_df = data_prep.split_train_val_test(features)
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    print(f"  train成功率={train_df['label'].mean():.3f}  val成功率={val_df['label'].mean():.3f}  test成功率={test_df['label'].mean():.3f}")

    print("[4/6] 前処理器を学習データにfitし、モデルを学習中...")
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[config.ALL_FEATURES])
    X_val = preprocessor.transform(val_df[config.ALL_FEATURES])
    X_test = preprocessor.transform(test_df[config.ALL_FEATURES])

    y_train = train_df["label"].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.float32)

    # クラス不均衡対策: sklearnの"balanced"と同じ考え方でサンプル重みを作る
    n_total = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_total - n_pos
    w_pos = n_total / (2.0 * n_pos)
    w_neg = n_total / (2.0 * n_neg)
    print(f"  クラス重み: 成功(label=1)={w_pos:.3f}, 失敗(label=0)={w_neg:.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)

    X_val_t = to_tensor(X_val).to(device)
    y_val_t = to_tensor(y_val).to(device)
    X_test_t = to_tensor(X_test).to(device)

    model = JobSuccessMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_auc = -1.0
    best_state = None
    patience_left = config.EARLY_STOP_PATIENCE

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            sample_weight = torch.where(yb == 1, w_pos, w_neg).to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, weight=sample_weight)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_sample_weight = torch.where(y_val_t == 1, w_pos, w_neg)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val_t, weight=val_sample_weight).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_probs)

        train_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        print(f"  epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = config.EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  早期終了(epoch {epoch}): val_aucがこれ以上改善しないため停止")
                break

    model.load_state_dict(best_state)
    model.eval()

    # --- 確率の較正 (Platt scaling) ---
    # クラス不均衡対策の重み付き損失は「順位付け(ROC-AUC)」の改善には有効だが、
    # 副作用として出力される確率の値そのものが偏ってしまう（失敗クラスの重みを
    # 大きくするほど、実際より低い成功確率を出しがちになる）。
    # 「成功確率〇%」とそのまま画面に表示するアプリである以上、この確率が
    # 実際の成功率に近いこと（較正されていること）が重要なため、
    # 検証(val)セットの生ロジットを使ってPlatt scaling（1次元のロジスティック
    # 回帰によるスケーリング）を行い、較正後の確率を最終出力とする。
    with torch.no_grad():
        val_logits_final = model(X_val_t).cpu().numpy().reshape(-1, 1)
    calibrator = LogisticRegression()
    calibrator.fit(val_logits_final, y_val)

    print("[5/6] テストセットで評価中...")
    with torch.no_grad():
        test_logits = model(X_test_t).cpu().numpy().reshape(-1, 1)
    test_probs_raw = 1 / (1 + np.exp(-test_logits.ravel()))  # 較正前（参考値）
    test_probs = calibrator.predict_proba(test_logits)[:, 1]  # 較正後（実際に使う値）
    test_pred = (test_probs >= 0.5).astype(int)

    # 較正の効果を確認するためのBrierスコア（小さいほど確率の精度が高い）
    brier_raw = float(brier_score_loss(y_test, test_probs_raw))
    brier_calibrated = float(brier_score_loss(y_test, test_probs))
    print(f"  Brierスコア: 較正前={brier_raw:.4f} 較正後={brier_calibrated:.4f}")

    metrics = {
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "test_success_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision_success": float(precision_score(y_test, test_pred, pos_label=1)),
        "recall_success": float(recall_score(y_test, test_pred, pos_label=1)),
        "f1_success": float(f1_score(y_test, test_pred, pos_label=1)),
        "precision_failure": float(precision_score(y_test, test_pred, pos_label=0)),
        "recall_failure": float(recall_score(y_test, test_pred, pos_label=0)),
        "f1_failure": float(f1_score(y_test, test_pred, pos_label=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "best_val_auc": float(best_val_auc),
        "mean_predicted_prob_raw": float(test_probs_raw.mean()),
        "mean_predicted_prob_calibrated": float(test_probs.mean()),
        "brier_score_raw": brier_raw,
        "brier_score_calibrated": brier_calibrated,
        "classification_report": classification_report(
            y_test, test_pred, target_names=["failure(0)", "success(1)"], output_dict=True
        ),
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print("[6/6] 学習済みモデル・前処理器・図・レポートを保存中...")

    # --- モデル本体 ---
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "hidden_dims": model.hidden_dims,
        },
        config.MODEL_PATH,
    )

    # --- 前処理器 ---
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    # --- 確率較正器 (Platt scaling) ---
    joblib.dump(calibrator, config.CALIBRATOR_PATH)

    # --- 選択肢（UIのドロップダウン用に学習データから実際の値を保存） ---
    options = {
        "freq_req": sorted(features["freq_req"].unique().tolist()),
    }
    joblib.dump(options, config.MODELS_DIR / "categorical_options.joblib")

    # --- 指標 ---
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # --- 図: 混同行列 ---
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["failure", "success"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (test)")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- 図: ROC曲線 ---
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, test_probs, ax=ax, name="MLP")
    ax.set_title("ROC Curve (test)")
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # --- 図: 学習曲線 ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(history["val_auc"], label="val_auc", color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend()
    axes[1].set_title("Validation ROC-AUC")
    fig.tight_layout()
    fig.savefig(fig_dir / "training_curve.png", dpi=150)
    plt.close(fig)

    # --- サンプルCSV ---
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_prep.save_sample_csv(features, config.DATA_DIR / "sample.csv")

    # --- レポートMarkdown ---
    _write_training_report(metrics, history)

    print("完了。")
    return metrics


def _write_training_report(metrics: dict, history: dict) -> None:
    report_path = config.DOCS_DIR / "training_report.md"
    cr = metrics["classification_report"]
    lines = [
        "# 学習・評価レポート",
        "",
        f"- 学習データ: {config.TRAIN_FILES[0]} 〜 {config.TRAIN_FILES[-1]}（{len(config.TRAIN_FILES)}ファイル）",
        f"- 検証データ（学習期間より未来）: {config.VAL_FILES}",
        f"- テストデータ（検証期間よりさらに未来）: {config.TEST_FILES}",
        f"- train件数: {metrics['n_train']:,} / val件数: {metrics['n_val']:,} / test件数: {metrics['n_test']:,}",
        f"- testデータの成功率（クラス比）: {metrics['test_success_rate']:.3f}",
        f"- 使用した特徴量: {config.ALL_FEATURES}",
        "",
        "使用する特徴量は、ジョブ投入時点で確実に分かる情報だけに限定して"
        "いる（特徴量の選定方針は docs/feature_leakage.md を参照）。",
        "",
        "## テストセットでの評価指標",
        "",
        "| 指標 | 値 |",
        "|---|---|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
        f"| Precision (success) | {metrics['precision_success']:.4f} |",
        f"| Recall (success) | {metrics['recall_success']:.4f} |",
        f"| F1 (success) | {metrics['f1_success']:.4f} |",
        f"| Precision (failure) | {metrics['precision_failure']:.4f} |",
        f"| Recall (failure) | {metrics['recall_failure']:.4f} |",
        f"| F1 (failure) | {metrics['f1_failure']:.4f} |",
        "",
        "failure（失敗）はクラス不均衡で少数派のため、Accuracyだけでなく"
        "failureクラスのPrecision/Recallも重視して評価している。",
        "",
        "## 確率の較正（Platt scaling）",
        "",
        "クラス不均衡対策の重み付き損失は順位付け（ROC-AUC）の改善には有効だが、"
        "副作用として出力される確率の値そのものが歪む（実際より低い成功確率を"
        "出しがちになる）。本アプリは「成功確率〇%」をそのまま画面に表示する"
        "ため、検証(val)セットの生ロジットを使ってPlatt scaling"
        "（1次元のロジスティック回帰によるスケーリング）を行い、"
        "較正後の確率を最終出力としている（`src/models/calibrator.joblib`）。",
        "",
        "| 指標 | 較正前 | 較正後 |",
        "|---|---|---|",
        f"| テストセットでの平均予測確率 | {metrics['mean_predicted_prob_raw']:.3f} | {metrics['mean_predicted_prob_calibrated']:.3f} |",
        f"| Brierスコア（小さいほど良い） | {metrics['brier_score_raw']:.4f} | {metrics['brier_score_calibrated']:.4f} |",
        "",
        f"（参考: テストセットの実際の成功率は {metrics['test_success_rate']:.3f}）",
        "",
        "## 混同行列・ROC曲線・学習曲線",
        "",
        "![confusion matrix](figures/confusion_matrix.png)",
        "",
        "![roc curve](figures/roc_curve.png)",
        "",
        "![training curve](figures/training_curve.png)",
        "",
        "## sklearn classification_report",
        "",
        "```",
        json.dumps(cr, indent=2, ensure_ascii=False),
        "```",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    train_one_run()
