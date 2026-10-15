"""Build <team>_submission.zip in the layout the organisers require (stdlib only).

    <team>_submission.zip
    ├── output/matching_results.tsv, output/candidate_pairs.tsv
    ├── code/business_entity_resolution/   (src/, README.md, requirements.txt)
    └── Documentation_template.md

Run from the project root (the directory with dataset/ and output/):

    python code/business_entity_resolution/src/tools/make_submission_zip.py --team <team_name>
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]           # code/business_entity_resolution
SKIP_DIRS = {"__pycache__", ".pytest_cache", "build", "dist"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SKIP_FILES = {".DS_Store"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name: writes <team>_submission.zip")
    ap.add_argument("--root", default=".", help="project root (contains output/)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    out = Path(a.out) if a.out else root / f"{a.team}_submission.zip"
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(root / "output" / f, f"output/{f}")
        for base, dirs, files in os.walk(CODE):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
            for f in files:
                p = Path(base) / f
                if p.suffix in SKIP_SUFFIXES or f in SKIP_FILES:
                    continue
                z.write(p, "code/business_entity_resolution/" + p.relative_to(CODE).as_posix())
        z.write(root / "Documentation_template.md", "Documentation_template.md")
    os.replace(tmp, out)
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        names = z.namelist()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(names)} files); "
          f"CRC check: {'OK' if bad is None else 'FAILED on ' + bad}")


if __name__ == "__main__":
    main()
