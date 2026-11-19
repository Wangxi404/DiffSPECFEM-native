"""Tests for the Marmousi-II example."""
import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffspecfem import native
from diffspecfem.inversion import ASSET_SHA256, Inversion, optimizer_step, stage_for_update, validate_inputs
from diffspecfem.soft_dtw_objective import soft_dtw_divergence
from scripts import agl_marmousi, build_specfem
from scripts.prepare_marmousi_inputs import prepare_inputs


def test_agl_download_uses_checked_http_fallback_and_warm_cache(tmp_path, monkeypatch):
    name = "vp_marmousi-ii.segy.gz"
    url = "https://www.agl.uh.edu/downloads/" + name
    content = b"sample-fixture"
    record = dict(url=url, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    calls = []

    def response(address, timeout):
        calls.append(address)
        if address == url:
            raise OSError("connection refused")
        return io.BytesIO(content)

    monkeypatch.setattr(agl_marmousi, 'urlopen', response)
    for _ in range(2):
        path = agl_marmousi.source_file(tmp_path, name, record, download=True)
        assert path.read_bytes() == content
    assert calls == [url, "http://www.agl.uh.edu/downloads/" + name]


@pytest.mark.parametrize("component,timestamp,second_response", [
    ("vp", "20070821164514", "truncated"),
    ("vs", "20070821164456", "wrong-hash"),
    ("density", "20070821164444", "interrupted"),
])
def test_agl_uses_verified_original_url_snapshot(tmp_path, monkeypatch, component,
                                                 timestamp, second_response):
    name = f"{component}_marmousi-ii.segy.gz"
    http = "http://www.agl.uh.edu/downloads/" + name
    https = http.replace("http:", "https:", 1)
    snapshot = f"https://web.archive.org/web/{timestamp}id_/{http}"
    content = b"sample-fixture"
    record = dict(url=https, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    calls = []

    def response(address, timeout):
        calls.append((address, timeout))
        if address == https:
            return io.BytesIO(b"<html>Not available</html>")
        if address == http:
            if second_response == "interrupted":
                raise IncompleteRead(partial=b"sample")
            body = content[:-1] if second_response == "truncated" else content.upper()
            return io.BytesIO(body)
        assert address == snapshot
        return io.BytesIO(content)

    monkeypatch.setattr(agl_marmousi, "urlopen", response)
    for _ in range(2):
        assert agl_marmousi.source_file(tmp_path, name, record, download=True).read_bytes() == content
    assert calls == [(https, 30), (http, 30), (snapshot, 120)]


def test_agl_connection_failure_leaves_cache_empty_and_explains_offline_input(tmp_path, monkeypatch):
    name = "vp_marmousi-ii.segy.gz"
    record = dict(url="https://www.agl.uh.edu/downloads/" + name, bytes=4,
                  sha256=hashlib.sha256(b"good").hexdigest())

    def unavailable(address, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(agl_marmousi, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="--source-dir"):
        agl_marmousi.source_file(tmp_path, name, record, download=True)
    assert list(tmp_path.iterdir()) == []


def test_agl_download_rejects_mismatched_bytes_before_caching(tmp_path, monkeypatch):
    name = "vs_marmousi-ii.segy.gz"
    good = b"good"
    record = dict(url="https://www.agl.uh.edu/downloads/" + name, bytes=len(good),
                  sha256=hashlib.sha256(good).hexdigest())
    monkeypatch.setattr(agl_marmousi, 'urlopen', lambda url, timeout: io.BytesIO(b"oops"))
    with pytest.raises(ValueError, match="Downloaded archive differs"):
        agl_marmousi.source_file(tmp_path, name, record, download=True)
    assert list(tmp_path.iterdir()) == []


def test_pinned_example_inputs_and_reference():
    training = validate_inputs(prepare_inputs())
    assert 'vp_true' not in training
    with np.load(prepare_inputs()/'model_fields.npz', allow_pickle=False) as model:
        for key in training:
            np.testing.assert_array_equal(training[key], model[key])
        assert model['depth_m'][-1] == 3515.
        assert model['sampling_depth_m'][-1] == 3500.
    reference = ROOT/'examples/marmousi2/reference'
    result = json.loads((reference/'result.json').read_text())
    historical = json.loads((ROOT/'docs/example/figures/manifest.json').read_text())
    assert result['completed_updates'] == historical['accepted_updates'] == 300
    assert result['input_edition'] == historical['input_edition'] == 'UH AGL download edition'
    assert result['final_model_sha256'] == historical['array_sha256']['vp_inverted']
    assert ASSET_SHA256['model_fields.npz'] == historical['input_npz_sha256']


@pytest.mark.parametrize("name", agl_marmousi.PACKAGES)
def test_paper_npz_writer_reproduces_original_archive_bytes(name):
    with np.load(prepare_inputs()/name, allow_pickle=False) as fields:
        output = io.BytesIO()
        agl_marmousi.write_paper_npz(output, dict(fields))
    assert hashlib.sha256(output.getvalue()).hexdigest() == ASSET_SHA256[name]


def test_material_interpolation_has_the_actual_transpose():
    rng = np.random.default_rng(53)
    x = np.r_[0., 8955., rng.uniform(0, 8955, 117)]
    z = np.r_[0., 3015., rng.uniform(0, 3015, 117)]
    mapping = native.mapping(x, z)
    model, cotangent = rng.normal(size=(68, 200)), rng.normal(size=len(x))
    lhs = np.sum(native.interp(model, mapping)*cotangent)
    rhs = np.sum(model*native.interp_T(cotangent, mapping))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(mapping[1].sum(axis=1), 1., atol=1e-15)


def test_record_sampling_has_the_actual_transpose():
    rng = np.random.default_rng(54)
    full, sampled = rng.normal(size=(2, 2, 7500)), rng.normal(size=(2, 2, 2500))
    lhs = np.sum(native.sample(full)*sampled)
    rhs = np.sum(full*native.sample_T(sampled))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-13, atol=1e-12)


def test_frequency_schedule_and_physical_update_constraints():
    config = json.loads((ROOT/'examples/marmousi2/config.json').read_text())
    assert [stage_for_update(i, config) for i in (0, 39, 40, 99, 100, 299)] == [3., 3., 4.5, 4.5, None, None]
    training = validate_inputs(prepare_inputs())
    initial = training['vp_initial'].astype(float)
    zeros = np.zeros_like(initial)
    taper = native.taper()
    gradient = np.cos(np.arange(initial.size).reshape(initial.shape)/271.)
    step = optimizer_step(initial, initial, zeros, zeros, zeros, gradient,
                          taper, training['vs_fixed'], 0, config)
    np.testing.assert_array_equal(step['vp'][taper == 0], initial[taper == 0])
    assert np.max(np.abs(step['vp']-initial)) <= 10.+1e-10
    assert np.all(step['vp']**2-4./3.*training['vs_fixed'].astype(float)**2 > 0.)


def test_debiased_objective_vjp_on_small_reference_problem():
    torch.manual_seed(55)
    observed = torch.randn(2, 8, dtype=torch.float64)
    predicted = (observed+.2*torch.randn_like(observed)).detach().requires_grad_(True)
    direction = torch.randn_like(predicted)
    def loss(value):
        return soft_dtw_divergence(value, observed, dt=.003, gamma=.1,
                                   backend='reference', scale=1., decimation=1)
    value = loss(predicted)
    derivative, = torch.autograd.grad(value, predicted)
    step = 1e-5
    centered = (loss(predicted.detach()+step*direction)-loss(predicted.detach()-step*direction))/(2*step)
    np.testing.assert_allclose(centered.item(), (derivative*direction).sum().item(), rtol=1e-7, atol=1e-9)
    assert abs(loss(observed).item()) < 1e-12


def plot_fixture(directory, input_hash=ASSET_SHA256['model_fields.npz']):
    with np.load(prepare_inputs()/'training.npz') as data:
        initial = data['vp_initial']
    np.savez(directory/'final_model.npz', vp=initial, initial=initial, iteration=2,
             input_model_sha256=input_hash, stop_reason='budget_exhaustion_2_updates')
    (directory/'metrics.jsonl').write_text('{"iteration":1,"loss":2,"stage":3}\n'
                                          '{"iteration":2,"loss":1,"stage":3}\n')


def test_incomplete_loss_history_cannot_be_presented_as_completed(tmp_path):
    from diffspecfem.plotting import render_run
    plot_fixture(tmp_path)
    (tmp_path/'metrics.jsonl').write_text('{"iteration":1,"loss":2,"stage":3}\n')
    with pytest.raises(ValueError, match='every accepted update'):
        render_run(ROOT, tmp_path)


def test_terminal_report_uses_newer_accepted_checkpoint_over_stale_result(tmp_path):
    from diffspecfem.plotting import render_run
    plot_fixture(tmp_path)
    with np.load(tmp_path/'final_model.npz') as archive:
        checkpoint = dict(archive)
    checkpoint['vp'] = checkpoint['vp'].copy()
    checkpoint['vp'][0, :2] = [1200, 6100]
    np.savez(tmp_path/'final_model.npz', **checkpoint)
    (tmp_path/'run.json').write_text(json.dumps({'configuration': {'objective': {'kind': 'Envelope misfit'}}}))
    (tmp_path/'result.json').write_text(json.dumps(dict(completed_updates=1,
                                    stop_reason='budget_exhaustion_1_updates', objective='old misfit')))
    report = render_run(ROOT, tmp_path)
    page = report.read_text()
    assert '2 accepted updates' in page and 'University of Houston AGL download edition' in page
    assert 'Envelope misfit' in page and 'old misfit' not in page
    manifest = json.loads((tmp_path/'figures/manifest.json').read_text())
    assert manifest['result_metadata_matches_checkpoint'] is False
    assert manifest['shared_vp_limits_km_s'] == [1.2, 6.1]


def test_plot_rejects_previous_data_edition(tmp_path):
    from diffspecfem.plotting import render_run
    plot_fixture(tmp_path, '69bd64d32107930b9a4a60597f5367b317094a6ddc30a73ade3ee10972b20269')
    with pytest.raises(ValueError, match='different datasets'):
        render_run(ROOT, tmp_path)
    assert not (tmp_path/'figures').exists()


def test_missing_build_is_rejected_before_creating_run(tmp_path):
    from diffspecfem.inversion import run_example
    config = json.loads((ROOT/'examples/marmousi2/config.json').read_text())
    with pytest.raises(ValueError, match='build-manifest.json missing'):
        run_example(ROOT, tmp_path/'run', config, source_dir=tmp_path, check_only=True)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('with_receipt', [True, False])
def test_resume_rejects_historical_or_missing_input_identity(tmp_path, monkeypatch, with_receipt):
    from diffspecfem import inversion
    config = json.loads((ROOT/'examples/marmousi2/config.json').read_text())
    runner = Inversion(ROOT, tmp_path, config)
    (tmp_path/'observations').mkdir()
    if with_receipt:
        (tmp_path/'run.json').write_text(json.dumps({'fingerprint_sha256':
            '345dd1f02f26223a96a23c4154e4890068111b9235c674fc4e6ecfdbf8e15f4b'}))
    monkeypatch.setattr(inversion, 'verify_build', lambda *args: {'binary_sha256': {}})
    with pytest.raises(ValueError, match='Run inputs'):
        runner.prepare(resume=True)
    assert not (tmp_path/'checkpoints').exists()
    assert not runner.active


def test_checkout_imports_only_the_pinned_snapshot(tmp_path, monkeypatch):
    original = tmp_path/'local'
    original.mkdir()

    def git(*args):
        return subprocess.check_output(['git', '-C', str(original), '-c', 'user.name=Test',
                '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgSign=false', *args],
                text=True).strip()

    git('init', '--quiet')
    git('commit', '--allow-empty', '-qm', 'ancestor')
    ancestor = git('rev-parse', 'HEAD')
    for name in ['configure', 'Makefile.in', 'EXAMPLES/Tromp2005_kernel/DATA/Par_file']:
        path = original/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    git('add', '.')
    git('commit', '-qm', 'snapshot')
    pin = git('rev-parse', 'HEAD')
    monkeypatch.setattr(build_specfem, 'COMMIT', pin)
    monkeypatch.setattr(build_specfem, 'TREE', git('rev-parse', 'HEAD^{tree}'))
    git('checkout', '-qb', 'unrelated')
    (original/'other-work').write_text('separate branch data')
    git('add', 'other-work')
    git('commit', '-qm', 'separate branch')
    other = git('rev-parse', 'HEAD')
    other_blob = git('rev-parse', 'HEAD:other-work')
    git('tag', 'unrelated-tag')
    git('checkout', '-q', '--detach', pin)

    source = original
    for stage in ('upstream', 'specfem2d'):
        target = tmp_path/stage
        build_specfem.checkout(source, target, tmp_path/'logs', stage)
        assert build_specfem.git(target, 'rev-list', 'HEAD') == pin
        assert not build_specfem.git(target, 'for-each-ref') and not build_specfem.git(target, 'remote')
        assert not (target/'.git/objects/info/alternates').exists()
        for excluded in (ancestor, other, other_blob):
            assert subprocess.run(['git', '-C', str(target), 'cat-file', '-e', excluded],
                                  capture_output=True).returncode != 0
        source = target


def test_build_identity_is_bound_to_pinned_sources_and_executables(tmp_path, monkeypatch):
    source = tmp_path/'build'
    (source/'bin').mkdir(parents=True)
    patcher = build_specfem.load_patch()
    expected = {}
    for name in patcher.FILES:
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        expected[name] = build_specfem.sha(path)
    monkeypatch.setattr(patcher, 'FILES', {name: ('unused', value) for name, value in expected.items()})
    monkeypatch.setattr(build_specfem, 'load_patch', lambda: patcher)
    answers = {'--show-toplevel': str(source.resolve()), 'HEAD': build_specfem.COMMIT,
               'HEAD^{tree}': build_specfem.TREE}
    monkeypatch.setattr(build_specfem, 'git', lambda repo, *args:
                        answers[args[-1]] if args[0] == 'rev-parse' else '\n'.join(expected))
    binaries = {}
    for name in ('xmeshfem2D', 'xspecfem2D'):
        path = source/'bin'/name
        path.write_text(name)
        path.chmod(0o755)
        binaries[name] = build_specfem.sha(path)
    receipt = source/'build-manifest.json'
    manifest = dict(status='built', upstream_commit=build_specfem.COMMIT, upstream_tree=build_specfem.TREE,
                    native_precision='float64', configure_command=build_specfem.CONFIGURE,
                    patch_sha256=patcher.PATCH_SHA256, instrumented_source_sha256=expected,
                    binary_sha256=binaries)
    receipt.write_text(json.dumps(manifest))
    elsewhere = tmp_path/'copied-bin'
    shutil.copytree(source/'bin', elsewhere)
    assert build_specfem.verify_build(source, elsewhere) == manifest
    (elsewhere/'xspecfem2D').write_text('foreign executable')
    with pytest.raises(ValueError, match='executable differs'):
        build_specfem.verify_build(source, elsewhere)
    receipt.write_text(json.dumps(dict(manifest, instrumented_source_sha256={})))
    with pytest.raises(ValueError, match='build manifest does not match'):
        build_specfem.verify_build(source)
