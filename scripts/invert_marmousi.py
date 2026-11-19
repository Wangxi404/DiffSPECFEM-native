#!/usr/bin/env python3
"""Run the supplied native P-SV Marmousi-II inversion and final visualization."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8, help='Independent native CPU shot workers (capped by allowed CPUs)')
    parser.add_argument('--device', default='cpu', help='Data-objective PyTorch device: cpu or cuda:0; propagation remains native CPU')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/marmousi2')
    parser.add_argument('--config', type=Path, default=ROOT/'examples/marmousi2/config.json')
    parser.add_argument('--source-dir', type=Path, default=ROOT/'build/specfem2d')
    parser.add_argument('--binary-dir', type=Path, help='Default: SOURCE_DIR/bin')
    parser.add_argument('--max-updates', type=int, default=300, help='300 for the complete example; fewer updates are a labelled smoke run')
    parser.add_argument('--resume', action='store_true', help='Explicitly resume the saved model, moments and global update counter')
    parser.add_argument('--check-only', action='store_true', help='Build native layout and check map/sampling transposes and data VJP, without inversion')
    parser.add_argument('--check-gradient', action='store_true', help='Also verify a single-shot native model directional derivative before inversion')
    args = parser.parse_args()
    from diffspecfem.inversion import run_example
    config = json.loads(args.config.read_text())
    run_example(ROOT, args.output, config, workers=args.workers, device=args.device,
                maximum_updates=args.max_updates, resume=args.resume,
                check_only=args.check_only, check_gradient=args.check_gradient,
                source_dir=args.source_dir, binary_dir=args.binary_dir)


if __name__ == '__main__':
    main()
