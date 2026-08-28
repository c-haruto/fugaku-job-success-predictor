"""
冨岳ジョブ成功率予測Webアプリ バックエンド（FastAPI）。

起動方法:
    uvicorn src.main:app --reload

ブラウザで http://127.0.0.1:8000/ を開くとフォームが表示される。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .inference import Predictor, now_hour_dow

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="冨岳ジョブ成功率予測")

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


class PredictRequest(BaseModel):
    n_nodes: int = Field(..., ge=1, le=200000, description="要求ノード数")
    elapse_hours: float = Field(..., ge=0.01, le=200, description="要求経過時間上限（時間）")
    freq_req: str = Field(..., description="要求周波数")
    submit_hour: int | None = Field(None, ge=0, le=23, description="投入予定時刻（0-23時、未指定なら現在時刻）")
    submit_dow: int | None = Field(None, ge=0, le=6, description="投入予定曜日（0=月,6=日、未指定なら現在の曜日）")
    mem_limit_gib: float | None = Field(None, ge=0.1, le=256, description="要求メモリ上限（GiB、任意。未指定なら上限なし扱い）")


class PredictResponse(BaseModel):
    success_probability: float
    failure_probability: float
    label: str
    message: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/options")
def options():
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return predictor.get_options()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    default_hour, default_dow = now_hour_dow()
    submit_hour = req.submit_hour if req.submit_hour is not None else default_hour
    submit_dow = req.submit_dow if req.submit_dow is not None else default_dow

    result = predictor.predict(
        n_nodes=req.n_nodes,
        elapse_hours=req.elapse_hours,
        freq_req=req.freq_req,
        submit_hour=submit_hour,
        submit_dow=submit_dow,
        mem_limit_gib=req.mem_limit_gib,
    )

    pct = result.success_probability * 100
    if pct >= 90:
        message = "過去の傾向から見て、成功する可能性が高いジョブです。"
    elif pct >= 70:
        message = "比較的成功しやすい条件ですが、念のため設定を見直すと安心です。"
    elif pct >= 40:
        message = "失敗する可能性も低くありません。要求条件を見直すことをおすすめします。"
    else:
        message = "失敗する可能性が高い条件です。ノード数・実行時間などの見直しを推奨します。"

    return PredictResponse(
        success_probability=result.success_probability,
        failure_probability=result.failure_probability,
        label=result.label,
        message=message,
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/assets", StaticFiles(directory=config.PROJECT_ROOT / "assets"), name="assets")
