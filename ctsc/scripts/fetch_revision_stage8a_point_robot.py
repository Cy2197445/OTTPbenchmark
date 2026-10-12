'Fetch the one preselected FISOR Point Robot file with a disk safety guard.'

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import requests


SHARE = "https://cloud.tsinghua.edu.cn/d/162d6fe92bde43e28676/"
TOKEN = "162d6fe92bde43e28676"
NAME = "point_robot-expert-random-100k.hdf5"
EXPECTED_BYTES = 10_904_096
FLOOR_BYTES = 10 * 1024**3
POST_DOWNLOAD_RESERVE_BYTES = 512 * 1024**2


def check_capacity(destination: Path, payload_bytes: int) -> dict[str, int]:
    'Reject when raw file plus a 512 MiB later-work reserve breaches 10 GiB.'
    free = shutil.disk_usage(destination.parent).free
    required = FLOOR_BYTES + POST_DOWNLOAD_RESERVE_BYTES + payload_bytes
    if free < required:
        raise RuntimeError(f"Insufficient disk capacity: free={free}, required={required}")
    return {"free_before_bytes": free, "required_before_bytes": required}


def fetch(destination: Path) -> dict[str, object]:
    destination = destination.resolve()
    if destination.name != NAME:
        raise ValueError(f"Expected destination filename {NAME}")
    if destination.exists() or destination.with_suffix(destination.suffix + ".partial").exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    capacity = check_capacity(destination, EXPECTED_BYTES)

    session = requests.Session()
    root = session.get(SHARE, timeout=20)
    root.raise_for_status()
    listing_url = f"https://cloud.tsinghua.edu.cn/api/v2.1/share-links/{TOKEN}/dirents/"
    listing_response = session.get(listing_url, params={"path": "/"}, timeout=20)
    listing_response.raise_for_status()
    listing = listing_response.json()["dirent_list"]
    if len(listing) != 1 or listing[0]["file_name"] != NAME or int(listing[0]["size"]) != EXPECTED_BYTES:
        raise RuntimeError(f"Official share listing changed: {listing}")

    params = {"p": f"/{NAME}", "dl": "1"}
    file_url = SHARE + "files/"
    head = session.head(file_url, params=params, allow_redirects=True, timeout=20)
    head.raise_for_status()
    if int(head.headers.get("content-length", -1)) != EXPECTED_BYTES:
        raise RuntimeError("Official file Content-Length differs from locked listing")

    partial = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    count = 0
    with session.get(file_url, params=params, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        with partial.open("xb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                count += len(chunk)
                if count > EXPECTED_BYTES:
                    raise RuntimeError("Download exceeded locked size")
                target.write(chunk)
                digest.update(chunk)
    if count != EXPECTED_BYTES:
        raise RuntimeError(f"Incomplete download: {count} != {EXPECTED_BYTES}")
    if digest.hexdigest() != "20773a5530cbda57644adac7226812cf492593a617735b4685a969131d37c470":
        raise RuntimeError("Downloaded file SHA256 differs from the recorded source")
    partial.rename(destination)
    free_after = shutil.disk_usage(destination.parent).free
    if free_after < FLOOR_BYTES:
        raise RuntimeError("disk safety floor breached after download")
    return {
        "source_share": SHARE,
        "source_file_path": f"/{NAME}",
        "destination": str(destination),
        "bytes": count,
        "sha256": digest.hexdigest(),
        "free_after_bytes": free_after,
        **capacity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"G8A source fetch failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
