"""Download the UH AGL Marmousi-II files and prepare the paper's model locally."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.agl_marmousi import PACKAGES, default_cache, digest, ensure_packages


def prepare_inputs(metadata=None, *, source_dir=None, cache_dir=None, output=None):
    metadata = Path(metadata) if metadata is not None else ROOT / "assets/marmousi2d"
    profile = json.loads((metadata / "protocol.json").read_bytes())
    manifest = json.loads((metadata / "manifest.json").read_bytes())
    cache = Path(cache_dir) if cache_dir is not None else default_cache()
    output = Path(output) if output is not None else cache / "native" / digest(metadata / "manifest.json")
    notices = ("DATA_ATTRIBUTION.txt", "DATA_TERMS.txt", "protocol.json", "manifest.json")

    def verify_existing():
        for name in notices:
            target = output / name
            if not target.is_file() or target.read_bytes() != (metadata / name).read_bytes():
                raise ValueError(f"Prepared directory contains missing or mismatched metadata: {target}")
        for name in PACKAGES:
            target = output / name
            if not target.is_file() or digest(target) != manifest["files"][name]["sha256"]:
                raise ValueError(f"Prepared inputs differ from the paper reference: {target}")

    if output.exists():
        verify_existing()
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "inputs"
        ensure_packages(staging, profile, manifest, source_dir=source_dir, cache_dir=cache)
        for name in notices:
            shutil.copyfile(metadata / name, staging / name)
        try:
            staging.rename(output)
        except OSError:
            if not output.is_dir():
                raise
            verify_existing()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="use existing UH AGL .segy.gz files offline")
    parser.add_argument("--cache-dir", type=Path, help="override the shared local AGL cache")
    parser.add_argument("--output", type=Path, help="export the prepared inputs to a separate directory")
    args = parser.parse_args()
    print(prepare_inputs(source_dir=args.source_dir, cache_dir=args.cache_dir, output=args.output))
