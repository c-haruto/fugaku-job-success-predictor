"""
F-DATA の parquet ファイルを読み込み、モデル学習用の特徴量を作るモジュール。

★データリーク防止の方針（詳細は config.py 冒頭のコメントを参照）
    ここで使う生カラムは config.SAFE_RAW_COLUMNS のみ（+ ターゲット列）。
    いずれもジョブ投入時点（実行開始前）に値が確定している情報のみである。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_raw_parquets(filenames: list[str], raw_dir=None) -> pd.DataFrame:
    """指定したparquetファイル群から必要な列だけを読み込んで結合する。"""
    raw_dir = raw_dir or config.RAW_DIR
    usecols = config.SAFE_RAW_COLUMNS + [config.TARGET_COLUMN]
    frames = []
    for fname in filenames:
        path = raw_dir / fname
        df = pd.read_parquet(path, columns=usecols)
        df["__source_file"] = fname
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """安全な生カラムから、モデル入力用の特徴量列を作る。"""
    df = df.copy()

    # 投入日時（タイムゾーン付き文字列 "2021-04-01 10:28:01+09"）をパース
    adt = pd.to_datetime(df["adt"], utc=True, errors="coerce")
    df["submit_hour"] = adt.dt.hour.astype(float)
    df["submit_dow"] = adt.dt.dayofweek.astype(float)
    df["adt"] = adt  # ソートのためdatetime型で保持

    nnumr = df["nnumr"].astype(float)
    elpl_hours = df["elpl"].astype(float) / 3600.0
    df["nnumr_log"] = np.log1p(nnumr)
    df["elpl_hours"] = elpl_hours

    # 要求ノード時間（ノード数×時間）: 単独の特徴量だけでは見えない
    # 「総リソース要求量」の目安になる派生特徴量
    df["node_hours_log"] = np.log1p(nnumr * elpl_hours)

    # メモリ上限: 指定しなかったジョブは巨大なセンチネル値になっているため、
    # 「指定したかどうか」のフラグと、指定時の実値(GiB, log1p変換)に分ける
    mszl = df["mszl"].astype(float)
    mszl_specified = (mszl < config.MSZL_SENTINEL_THRESHOLD).astype(float)
    mszl_gib = np.where(mszl_specified == 1, mszl / (1024**3), 0.0)
    df["mszl_specified"] = mszl_specified
    df["mszl_gib_log"] = np.log1p(mszl_gib)

    df["freq_req"] = df["freq_req"].astype(str)

    df["label"] = (df[config.TARGET_COLUMN] == config.POSITIVE_LABEL).astype(int)
    return df


def split_train_val_test(df_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    時系列を考慮した分割:
      - train: config.TRAIN_FILES 由来の全データ
      - val  : config.VAL_FILES 由来の全データ（trainより完全に未来）
      - test : config.TEST_FILES 由来の全データ（valよりさらに未来）
    """
    train_df = df_features[df_features["__source_file"].isin(config.TRAIN_FILES)].sort_values("adt")
    val_df = df_features[df_features["__source_file"].isin(config.VAL_FILES)].sort_values("adt")
    test_df = df_features[df_features["__source_file"].isin(config.TEST_FILES)].sort_values("adt")
    return train_df, val_df, test_df


def save_sample_csv(df_features: pd.DataFrame, path, n: int = 500) -> None:
    """レポート・動作確認用に、前処理後の特徴量の一部をサンプルCSVとして保存する。"""
    cols = config.ALL_FEATURES + ["label", "adt", "__source_file"]
    sample = df_features[cols].sample(n=min(n, len(df_features)), random_state=config.RANDOM_SEED)
    sample = sample.sort_values("adt")
    sample.to_csv(path, index=False, encoding="utf-8-sig")
