#!/usr/bin/env python3
"""Plot saved results without building or running SPECFEM2D."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffspecfem.plotting import render_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'runs/marmousi2')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(render_run(ROOT, args.run, args.output))


if __name__ == '__main__':
    main()
