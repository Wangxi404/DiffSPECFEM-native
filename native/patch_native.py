"""Apply the Marmousi-II I/O patch to the pinned SPECFEM2D source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PATCH_SHA256 = '955e00cc936db09f875e775f2b57f14a4a3db2b7c5f300040a9c912ac0d517f7'
PREVIOUS_PATCH_SHA256 = '0f71effbe68d3e09ae54baddb08bfa14a73312af87800256fd13244f87a17427'
PREVIOUS_LAYOUT_SHA256 = 'ea34b17fd889cd47dc2bd20f1b6d3b6f1803b4c159eca98128a16b2443ab0734'
FILES = {
    'src/specfem2D/save_model_files.f90': (
        'f65b091af4ad985ac2a28353c0e0f7944f49566d69132b234493f5bfa20b0e14',
        '50f94757193053ad32b4e43c13172b035736d040a4c032deed15f553a5eb9941'),
    'src/specfem2D/save_adjoint_kernels.f90': (
        '05026a68f68b756f17011dc19d518397da5a88028fdfa3e9feac8c51b5a497c3',
        '5bb86f33557b8c6ae47bad561766853cdc1ff7285a184ae67d42deeb02e6013f'),
    'src/specfem2D/compute_arrays_source.f90': (
        'e09f8dd4dac6da7c4fe3f440d75ad7bcf735b0a2650f1427491955c42b625462',
        'e9cf3988d0ff1b2ac337efef3a919f625cb18b1c7ddba2f4e21b99e750491308'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patched_files(patch_sha):
    hashes = {name: values[1] for name, values in FILES.items()}
    if patch_sha == PREVIOUS_PATCH_SHA256:
        # The original release used a different comment in the layout writer.
        hashes['src/specfem2D/save_model_files.f90'] = PREVIOUS_LAYOUT_SHA256
    elif patch_sha != PATCH_SHA256:
        raise ValueError('Unknown SPECFEM2D patch digest')
    return hashes


def apply(root):
    root = Path(root).resolve()
    patch = Path(__file__).resolve().with_name('extension.patch')
    if sha(patch) != PATCH_SHA256:
        raise ValueError('Unexpected checksum for extension.patch')
    actual = {name: sha(root/name) for name in FILES}
    after = patched_files(PATCH_SHA256)
    for known_patch in (PATCH_SHA256, PREVIOUS_PATCH_SHA256):
        if actual == patched_files(known_patch):
            return dict(status='already_applied', patch_sha256=known_patch, files=actual)
    if actual != {name: values[0] for name, values in FILES.items()}:
        raise ValueError('Native source does not match the pinned SPECFEM2D revision')
    top = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--show-toplevel'], text=True).strip()
    if Path(top).resolve() != root:
        raise ValueError('Patch target must be its own pinned SPECFEM2D Git checkout')
    for options in (['--check'], []):
        result = subprocess.run(['git', '-C', str(root), 'apply', *options, str(patch)],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Cannot apply the pinned native extension: '+result.stderr.strip())
    if {name: sha(root/name) for name in FILES} != after:
        raise ValueError('Patched SPECFEM2D source failed checksum verification')
    return dict(status='applied', patch_sha256=PATCH_SHA256, files=after)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print(json.dumps(apply(args.source)))
