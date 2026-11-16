"""
Self-contained LaTeX -> PDF builder for both the English (`main.tex` -> `ffmod_paper_en.pdf`)
and Hungarian (`main_hu.tex` -> `ffmod_paper_hu.pdf`) papers using the static `tectonic` XeTeX+BibTeX engine.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tarfile
import urllib.request
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path("/tmp/tectonic_cache")
TECTONIC_BIN = CACHE_DIR / "tectonic"
TECTONIC_URL = (
    "https://github.com/tectonic-typesetting/tectonic/releases/download/"
    "tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz"
)


def ensure_tectonic() -> Path:
    if TECTONIC_BIN.exists():
        return TECTONIC_BIN
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tgz_path = CACHE_DIR / "tectonic.tar.gz"
    print(f"[paper] Downloading standalone Tectonic XeTeX+BibTeX binary -> {TECTONIC_BIN}...")
    urllib.request.urlretrieve(TECTONIC_URL, str(tgz_path))
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extract("tectonic", path=str(CACHE_DIR))
    TECTONIC_BIN.chmod(TECTONIC_BIN.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return TECTONIC_BIN


def compile_tex(tectonic: Path, tex_name: str, out_pdf_name: str) -> Path:
    xdg_dir = CACHE_DIR / "xdg"
    env = os.environ.copy()
    env.setdefault("XDG_CACHE_HOME", str(xdg_dir))
    tex_path = PAPER_DIR / tex_name
    cmd = [str(tectonic), str(tex_path), "--outdir", str(PAPER_DIR)]
    if xdg_dir.exists():
        cmd.append("--only-cached")
    print(f"[paper] Compiling {tex_name} -> {out_pdf_name}...")
    subprocess.run(cmd, cwd=str(PAPER_DIR), env=env, check=True)
    built_pdf = PAPER_DIR / (Path(tex_name).stem + ".pdf")
    final_pdf = PAPER_DIR / out_pdf_name
    if built_pdf != final_pdf:
        shutil.move(str(built_pdf), str(final_pdf))
    size_kb = final_pdf.stat().st_size / 1024.0
    print(f"[paper] Built {final_pdf} ({size_kb:.1f} KB)")
    return final_pdf


def main() -> None:
    tectonic = ensure_tectonic()
    compile_tex(tectonic, "main.tex", "ffmod_paper_en.pdf")
    compile_tex(tectonic, "main_hu.tex", "ffmod_paper_hu.pdf")
    print("[paper] Both English and Hungarian PDFs built successfully!")


if __name__ == "__main__":
    main()
