#!/usr/bin/env python3
"""Build SPECFEM2D with the Marmousi-II inversion interface."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_URL = 'https://github.com/SPECFEM/specfem2d.git'
COMMIT = 'abf3196e3192af3343f2b402c82f34205577e672'
TREE = 'fe42896b23e4111933ba0c1d335dc7f57ec8737c'
CONFIGURE = ['./configure', '--with-mpi', '--enable-double-precision',
             'FC=gfortran', 'CC=gcc', 'MPIFC=mpif90']
PATCH = PROJECT/'native/extension.patch'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def check_source(source):
    source = Path(source).resolve()
    if git(source, 'rev-parse', '--show-toplevel') != str(source):
        raise ValueError(f'{source}: expected a complete SPECFEM2D Git checkout')
    if git(source, 'rev-parse', 'HEAD') != COMMIT or git(source, 'rev-parse', 'HEAD^{tree}') != TREE:
        raise ValueError(f'{source}: checkout must be at pinned commit {COMMIT}')
    if git(source, 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError(f'{source}: original source must have a clean working tree')
    for name in ('configure', 'Makefile.in', 'EXAMPLES/Tromp2005_kernel/DATA/Par_file'):
        if not (source/name).is_file():
            raise ValueError(f'{source}: incomplete original checkout; missing {name}')


def load_patch():
    spec = importlib.util.spec_from_file_location('native_patch', PROJECT/'native/patch_native.py')
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    return patcher


def verify_build(source, binary_dir=None):
    """Check a locally built solver before reuse, including relocated executables."""
    source = Path(source).resolve()
    receipt = source/'build-manifest.json'
    if not receipt.is_file():
        raise ValueError(f'{source}: build-manifest.json missing; use scripts/build_specfem.py to build the example')
    manifest = json.loads(receipt.read_text())
    patcher = load_patch()
    if not isinstance(manifest, dict) or sha(PATCH) != patcher.PATCH_SHA256:
        raise ValueError(f'{source}: build manifest or distributed patch is invalid')
    expected_sources = patcher.patched_files(manifest.get('patch_sha256'))
    expected_fields = dict(status='built', upstream_commit=COMMIT, upstream_tree=TREE,
                           native_precision='float64',
                           configure_command=CONFIGURE, instrumented_source_sha256=expected_sources)
    if any(manifest.get(key) != value for key, value in expected_fields.items()):
        raise ValueError(f'{source}: build manifest does not match the pinned SPECFEM2D extension')
    if (git(source, 'rev-parse', '--show-toplevel') != str(source) or
            git(source, 'rev-parse', 'HEAD') != COMMIT or git(source, 'rev-parse', 'HEAD^{tree}') != TREE):
        raise ValueError(f'{source}: expected the pinned SPECFEM2D checkout')
    # NativeSolver also reads the upstream Par_file and SOURCE to prepare the case.
    delta = git(source, 'diff', '--name-only', 'HEAD', '--', 'src',
                'EXAMPLES/Tromp2005_kernel/DATA/Par_file', 'EXAMPLES/Tromp2005_kernel/DATA/SOURCE')
    if set(delta.splitlines()) != set(expected_sources) or any(
            sha(source/name) != expected for name, expected in expected_sources.items()):
        raise ValueError(f'{source}: native source or example inputs differ from the tested extension')
    hashes = manifest.get('binary_sha256')
    if not isinstance(hashes, dict) or set(hashes) != {'xmeshfem2D', 'xspecfem2D'}:
        raise ValueError(f'{receipt}: missing the two native executable hashes')
    binary = (source/'bin' if binary_dir is None else Path(binary_dir)).resolve()
    for directory in dict.fromkeys([(source/'bin').resolve(), binary]):
        for name, expected in hashes.items():
            path = directory/name
            if not path.is_file() or not os.access(path, os.X_OK) or sha(path) != expected:
                raise ValueError(f'{path}: executable differs from the verified local build; use its bin directory')
    return manifest


def logged(command, cwd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('w') as stream:
        stream.write('Command: '+repr(command)+'\n')
        stream.flush()
        environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
        result = subprocess.run(command, cwd=cwd, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Build command failed with exit {result.returncode}; see {logfile}')


def checkout(source, target, logs, label):
    if target.exists():
        check_source(target)
        return
    logged(['git', 'init', '--quiet', str(target)], target.parent, logs/(label+'-init.log'))
    logged(['git', '-C', str(target), 'fetch', '--no-tags', '--depth=1',
            '--no-recurse-submodules', '--', str(source), COMMIT], target.parent, logs/(label+'-fetch.log'))
    logged(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'],
           target.parent, logs/(label+'-checkout.log'))
    check_source(target)


def build(build_dir=PROJECT/'build', source=None, jobs=8):
    if jobs < 1:
        raise ValueError('--jobs must be positive')
    build_dir = Path(build_dir).expanduser().resolve()
    upstream, native = build_dir/'upstream', build_dir/'specfem2d'
    manifest_path = native/'build-manifest.json'
    if native.exists():
        return verify_build(native)
    missing = [name for name in ('git', 'gfortran', 'gcc', 'mpif90', 'make') if not shutil.which(name)]
    if missing:
        raise RuntimeError('Required build tools not found: '+', '.join(missing)+
                           '. Install a GNU Fortran/C toolchain and an MPI Fortran development package.')
    build_dir.mkdir(parents=True, exist_ok=True)
    logs = build_dir/'logs'
    if source is not None:
        source = Path(source).expanduser().resolve()
        check_source(source)
    checkout(source if source is not None else SOURCE_URL, upstream, logs, 'upstream')
    checkout(upstream, native, logs, 'native')
    logged(CONFIGURE, native, logs/'configure.log')
    constants = (native/'setup/constants.h').read_text()
    if not re.search(r'CUSTOM_REAL\s*=\s*(?:SIZE_DOUBLE|8)\b', constants):
        raise ValueError('Configured SPECFEM2D is not double precision; no build attempted')
    patch_result = load_patch().apply(native)
    changed = set(git(native, 'diff', '--name-only', 'HEAD', '--', 'src').splitlines())
    if changed != set(patch_result['files']):
        raise ValueError('The instrumented source differs beyond the three tested hooks')
    logged(['make', '-j'+str(jobs), 'all'], native, logs/'make.log')
    binaries = {name: sha(native/'bin'/name) for name in ('xmeshfem2D', 'xspecfem2D')}
    manifest = dict(schema_version=1, status='built', created_utc=datetime.now(timezone.utc).isoformat(),
        upstream_url=SOURCE_URL, upstream_commit=COMMIT, upstream_tree=TREE,
        native_precision='float64', model_input_precision='float32; promoted at native material I/O',
        configure_command=CONFIGURE, build_jobs=jobs, patch_sha256=patch_result['patch_sha256'],
        instrumented_source_sha256=patch_result['files'], binary_sha256=binaries,
        fortran_record_marker_bytes=4, native_mpi_world_size_per_shot=1,
        propagation='Original SPECFEM2D forward, native adjoint and backward reconstruction',
        compiler_versions={name: subprocess.check_output([name, '--version'], text=True).splitlines()[0]
                           for name in ('gfortran', 'gcc', 'mpif90')})
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--source', type=Path,
                        help='Existing full clean checkout at the pinned commit (no network needed)')
    parser.add_argument('--build-dir', type=Path, default=PROJECT/'build')
    args = parser.parse_args()
    try:
        result = build(args.build_dir, args.source, args.jobs)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
