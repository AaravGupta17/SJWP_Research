"""
download_public_data.py — fetch the public real-world leak datasets
=====================================================================
Downloads into datasets/public/<name>/ (git-ignored). Safe to re-run:
finished files are skipped and verified by size (and hash when the
repository publishes one).

  hongkong   Tijani, Tariq, Zayed et al. (2022). Acoustic Based Data Acquisition
             for Leak Detection of Water Distribution Networks. Mendeley Data,
             doi:10.17632/hkn8mxcjyz.1. CC BY 4.0. Real buried networks, ~90 leak
             sites; hydrophones, noise loggers, MEMS accelerometers. ~257 MB.
  sheffield  Shekofteh, M.R. (2026). Acoustic data for the leakage experiments in
             the CID Lab, University of Sheffield. ORDA,
             doi:10.15131/shef.data.32229270.v1. CC BY 4.0. MDPE 63 mm pipe, two
             calibrated accelerometers. ~493 MB zip.
  dongguan   Wang, Mei, Zhan & Chen (2026). Acoustic data for 'Self-supervised
             acoustic leakage detection for water distribution systems'. Zenodo,
             doi:10.5281/zenodo.18631450. CC BY 4.0. 1 s clips. ~6 MB (RAR).

Usage (from the repo root):
    python scripts/download_public_data.py                 # all three
    python scripts/download_public_data.py dongguan hongkong
"""

import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "datasets" / "public"
# some repositories (Mendeley) reject Python's default user agent
HEADERS = {"User-Agent": "Mozilla/5.0 (research data download)"}


def _open(url: str, timeout: int):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout)


def _get_json(url: str):
    with _open(url, 60) as r:
        return json.load(r)


def _download(url: str, dest: Path, size: int = None, sha256: str = None, md5: str = None):
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  {dest.relative_to(OUT)}  ({(size or 0) / 1e6:.1f} MB)")
    with _open(url, 120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if size is not None and tmp.stat().st_size != size:
        raise IOError(f"size mismatch for {dest.name}: {tmp.stat().st_size} != {size}")
    for algo, want in (("sha256", sha256), ("md5", md5)):
        if want:
            h = hashlib.new(algo)
            with open(tmp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != want:
                raise IOError(f"{algo} mismatch for {dest.name}")
    tmp.replace(dest)


def _extract(archive: Path, into: Path):
    marker = into / f".extracted_{archive.name}"
    if marker.exists():
        return
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:   # .rar: bsdtar (built into Windows 10/11, libarchive on Linux/macOS)
        win_tar = Path(r"C:\Windows\System32\tar.exe")      # bsdtar; Git's GNU tar can't read RAR
        tar = shutil.which("bsdtar") or (str(win_tar) if win_tar.exists() else shutil.which("tar"))
        subprocess.run([tar, "-xf", str(archive), "-C", str(into)], check=True)
    marker.touch()


def hongkong():
    api = "https://data.mendeley.com/public-api/datasets/hkn8mxcjyz"
    meta = _get_json(api)
    folders = {f["id"]: f for f in _get_json(api + "/folders/1")}

    def path_of(fid):
        parts = []
        while fid in folders:
            parts.append(folders[fid]["name"])
            fid = folders[fid].get("parent_id")
        return Path(*reversed(parts)) if parts else Path()

    root = OUT / "hongkong"
    for f in meta["files"]:
        c = f["content_details"]
        _download(c["download_url"], root / path_of(f["folder_id"]) / f["filename"],
                  size=f["size"], sha256=c.get("sha256_hash"))


def sheffield():
    meta = _get_json("https://api.figshare.com/v2/articles/32229270")
    root = OUT / "sheffield"
    for f in meta["files"]:
        dest = root / f["name"]
        _download(f["download_url"], dest, size=f["size"], md5=f.get("supplied_md5"))
        if dest.suffix.lower() == ".zip":
            _extract(dest, root / "extracted")


def dongguan():
    meta = _get_json("https://zenodo.org/api/records/18631450")
    root = OUT / "dongguan"
    for f in meta["files"]:
        dest = root / f["key"]
        md5 = f["checksum"].split(":", 1)[1] if f.get("checksum", "").startswith("md5:") else None
        _download(f["links"]["self"], dest, size=f["size"], md5=md5)
        _extract(dest, root / "extracted")


DATASETS = {"hongkong": hongkong, "sheffield": sheffield, "dongguan": dongguan}

if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS)
    for n in names:
        print(f"\n{n}")
        DATASETS[n]()
    print(f"\nDone: {OUT}")
