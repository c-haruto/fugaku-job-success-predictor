"""
F-DATA (Zenodo, DOI: 10.5281/zenodo.11467483) の月次parquetファイルを
data/raw/ にダウンロードするスクリプト。

config.py で指定した TRAIN_FILES / VAL_FILES / TEST_FILES に含まれる
ファイルだけを取得する。Zenodo側の帯域制限が1接続あたりに掛かっている
ようなので、複数ファイルを並列ダウンロードすることで全体のスループットを
上げている（実測: 1接続=約0.6MB/s、16並列=約3.6MB/s）。

使い方:
    python -m src.download_data
    python -m src.download_data --workers 8   # 並列数を変える場合
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config

DEFAULT_WORKERS = 16


def _download_one(filename: str) -> str:
    dest = config.RAW_DIR / filename
    if dest.exists():
        return f"[skip] {filename} は既に存在します ({dest.stat().st_size / 1e6:.1f} MB)"

    url = f"{config.ZENODO_FILES_BASE_URL}/{filename}/content"
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp_dest)
        tmp_dest.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        return f"[done] {filename} ({size_mb:.1f} MB)"
    except Exception as e:
        if tmp_dest.exists():
            tmp_dest.unlink()
        return f"[error] {filename}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="並列ダウンロード数")
    args = parser.parse_args()

    files = list(dict.fromkeys(config.TRAIN_FILES + config.VAL_FILES + config.TEST_FILES))
    print(f"ダウンロード対象: {len(files)}ファイル")
    print(f"並列数: {args.workers}")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_download_one, f): f for f in files}
        for future in as_completed(futures):
            done += 1
            print(f"({done}/{len(files)}) {future.result()}")
            sys.stdout.flush()

    print("完了。")


if __name__ == "__main__":
    main()
