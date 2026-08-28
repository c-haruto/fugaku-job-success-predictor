"""
学習済みモデルを読み込み、投入前のジョブ情報から成功確率を予測するモジュール。
FastAPI アプリ（main.py）から呼び出される。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch

from . import config
from .model import JobSuccessMLP


@dataclass
class PredictionResult:
    success_probability: float
    failure_probability: float
    label: str  # "success" or "failure"（0.5を閾値とした予測ラベル）


class Predictor:
    """モデル・前処理器・確率較正器をまとめて保持し、推論を行うクラス。"""

    def __init__(self):
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"学習済みモデルが見つかりません: {config.MODEL_PATH}\n"
                "先に `python -m src.train` を実行してモデルを学習してください。"
            )

        checkpoint = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=False)
        self.model = JobSuccessMLP(input_dim=checkpoint["input_dim"], hidden_dims=checkpoint["hidden_dims"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        # 学習時に検証(val)セットで較正したPlatt scaling器。
        # モデルの生ロジットをそのままsigmoidした確率は、クラス不均衡対策の
        # 重み付き損失の影響で歪んでいるため、必ずこれを通してから表示する。
        self.calibrator = joblib.load(config.CALIBRATOR_PATH)

        self.options: dict = joblib.load(config.MODELS_DIR / "categorical_options.joblib")

    def get_options(self) -> dict:
        return self.options

    def predict(
        self,
        n_nodes: int,
        elapse_hours: float,
        freq_req: str,
        submit_hour: int,
        submit_dow: int,
        mem_limit_gib: float | None = None,
    ) -> PredictionResult:
        mszl_specified = 1.0 if mem_limit_gib is not None else 0.0
        mszl_gib_log = np.log1p(mem_limit_gib) if mem_limit_gib is not None else 0.0

        row = {
            "nnumr_log": np.log1p(max(n_nodes, 0)),
            "elpl_hours": elapse_hours,
            "node_hours_log": np.log1p(max(n_nodes, 0) * elapse_hours),
            "mszl_specified": mszl_specified,
            "mszl_gib_log": mszl_gib_log,
            "submit_hour": submit_hour,
            "submit_dow": submit_dow,
            "freq_req": freq_req,
        }

        df = pd.DataFrame([row])[config.ALL_FEATURES]
        x = self.preprocessor.transform(df)
        x_t = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            logit = self.model(x_t).item()
        prob_success = float(self.calibrator.predict_proba([[logit]])[0, 1])

        label = "success" if prob_success >= 0.5 else "failure"
        return PredictionResult(
            success_probability=prob_success,
            failure_probability=1.0 - prob_success,
            label=label,
        )


def now_hour_dow() -> tuple[int, int]:
    now = datetime.now()
    return now.hour, now.weekday()
