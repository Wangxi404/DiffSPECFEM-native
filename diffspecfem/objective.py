"""Per-shot soft-DTW divergence on traces sampled every 3 ms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .soft_dtw_objective import PreparedSoftDTWDivergence, backend_provenance

PROFILE_ID = "marmousi-ii-fast40-softdtw-v1"
DT = 0.003
NT = 2500
STAGES = (3.0, 4.5, None)


def stage_cutoff(stage):
    """Accept a cutoff (3, 4.5, None), or explicit strings stage0/1/2."""
    if isinstance(stage, str):
        stage = {"stage0": 3.0, "stage1": 4.5, "stage2": None,
                 "3": 3.0, "4.5": 4.5, "none": None, "full": None}[stage]
    if stage not in STAGES:
        raise ValueError("stage must be cutoff 3., 4.5, None, or stage0/1/2")
    return stage


def lowpass(data, cutoff, transition=0.5, dt=DT):
    """Zero-phase FFT low-pass filter with one record of padding on each side."""
    if cutoff is None:
        return data
    nt = data.shape[-1]
    frequencies = torch.fft.rfftfreq(3 * nt, dt, dtype=data.dtype,
                                    device=data.device)
    position = ((frequencies - (cutoff - transition)) /
                (2 * transition)).clamp(0, 1)
    response = 0.5 + 0.5 * torch.cos(torch.pi * position)
    padded = F.pad(data, (nt, nt))
    filtered = torch.fft.irfft(torch.fft.rfft(padded, dim=-1) * response,
                               n=3 * nt, dim=-1)
    return filtered[..., nt:2 * nt]


class Fast40Objective:
    """Cache one observed shot at one frequency stage for first-order VJPs."""
    def __init__(self, observed, stage=None, *, backend="pysdtw", device="cpu"):
        self.stage = stage_cutoff(stage)
        self.device = torch.device(device)
        self.backend = backend
        obs = torch.as_tensor(np.asarray(observed), dtype=torch.float64,
                              device=self.device).detach()
        self.device = obs.device  # resolve bare 'cuda' to its actual CUDA index
        if obs.ndim < 1 or obs.shape[-1] != NT:
            raise ValueError("Fast40 observation must have final time axis 2500")
        if not bool(torch.isfinite(obs).all()):
            raise ValueError("Observed data must be finite")
        self.shape = tuple(obs.shape)
        self.observed_sha256 = hashlib.sha256(
            np.ascontiguousarray(obs.cpu().numpy(), dtype='<f8').tobytes()).hexdigest()
        filtered = lowpass(obs, self.stage)
        self.scale = float(filtered.square().mean().sqrt().item())
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Stage-filtered observation RMS must be positive")
        self.prepared = PreparedSoftDTWDivergence(
            filtered, DT, gamma=0.1, scale=self.scale, decimation=8,
            anti_alias_cutoff_hz=10.0, anti_alias_transition_hz=2.0,
            backend=backend, bandwidth=None)

    def tensor_loss(self, predicted):
        """Apply every prediction-dependent filter in its original AD graph."""
        if tuple(predicted.shape) != self.shape:
            raise ValueError("Prediction/observation shape mismatch")
        if predicted.dtype != torch.float64 or predicted.device != self.device:
            raise ValueError("Prediction must be float64 on objective device")
        return self.prepared(lowpass(predicted, self.stage))

    def value(self, predicted):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(predicted), dtype=torch.float64,
                                device=self.device)
            return float(self.tensor_loss(x).item())

    def loss_and_cotangent(self, predicted):
        leaf = torch.as_tensor(np.asarray(predicted), dtype=torch.float64,
                               device=self.device).detach().requires_grad_(True)
        loss = self.tensor_loss(leaf)
        cotangent, = torch.autograd.grad(loss, leaf)
        if not bool(torch.isfinite(cotangent).all()):
            raise FloatingPointError("Nonfinite objective cotangent")
        return float(loss.detach().item()), cotangent.detach().cpu().numpy().copy()

    def configuration(self):
        return dict(profile=PROFILE_ID, observation_sha256=self.observed_sha256,
                    input_shape=self.shape, stage_cutoff_hz=self.stage,
                    stage_transition_hz=0.5, fixed_observation_rms=self.scale,
                    trace_components="All leading axes, uniformly weighted",
                    shot_normalization="One shot only; caller sums /40 once",
                    native_sampling="Outside this data-only boundary",
                    **self.prepared.configuration())


_CACHE = {}


def loss_and_cotangent(pred, obs, stage=None, *, backend="pysdtw", device="cpu"):
    """Return loss and data derivative, reusing the prepared observation."""
    a = np.ascontiguousarray(obs, dtype='<f8')
    key = (hashlib.sha256(a.tobytes()).hexdigest(), a.shape, stage_cutoff(stage),
           backend, str(device))
    if key not in _CACHE:
        _CACHE[key] = Fast40Objective(a, stage, backend=backend, device=device)
    return _CACHE[key].loss_and_cotangent(pred)


def clear_cache():
    _CACHE.clear()


def validate(output, *, device="cpu", backend="pysdtw", traces=3):
    """Check the data derivative by centered differences at each stage."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261113)
    t = DT * np.arange(NT)
    obs = []
    pred = []
    for i in range(traces):
        env = np.exp(-((t - 1.7 - 0.3 * i) / 0.65) ** 2)
        obs.append(env * (np.sin(2 * np.pi * (2.2 + i) * t)
                          + 0.13 * np.cos(2 * np.pi * 8.5 * t)))
        pred.append(np.roll(obs[-1], 7 + 2 * i) * 0.91
                    + 0.03 * np.sin(2 * np.pi * 6 * t) * env)
    obs, pred = np.asarray(obs), np.asarray(pred)
    direction = rng.standard_normal(pred.shape)
    # Use a band-limited direction with unit RMS at every step size.
    direction = lowpass(torch.from_numpy(direction), 7., 1.).numpy()
    direction /= np.sqrt(np.mean(direction ** 2))
    results = []
    for stage in STAGES:
        obj = Fast40Objective(obs, stage, device=device, backend=backend)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        loss, v = obj.loss_and_cotangent(pred)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        dot = float(np.sum(direction * v))
        rows = []
        for h in (0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003):
            plus, minus = obj.value(pred + h * direction), obj.value(pred - h * direction)
            derivative = (plus - minus) / (2 * h)
            rows.append(dict(epsilon=h, plus=plus, minus=minus,
                             centered_derivative=derivative, ad_dot=dot,
                             absolute_error=abs(derivative - dot),
                             relative_error=abs(derivative - dot) / max(abs(dot), 1e-30),
                             forward_taylor_error=abs(plus - loss - h * dot)))
        results.append(dict(configuration=obj.configuration(), value=loss,
                            vjp_norm=float(np.linalg.norm(v)), ad_dot=dot,
                            vjp_seconds=seconds, sweep=rows,
                            passed=abs(dot) > 1e-8 and
                            min(r['relative_error'] for r in rows) < 3e-5))
    report = dict(kind="data-only; does not qualify native sampling/adjoint",
                  backend=backend, device=device, seed=20261113,
                  backend_provenance=backend_provenance() if backend == 'pysdtw' else None,
                  numpy=np.__version__, torch=torch.__version__, stages=results,
                  passed=all(r['passed'] for r in results))
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(str(Path(output).with_suffix('.npz')), observed=obs,
                        prediction=pred, direction=direction,
                        cotangent_stage0=Fast40Objective(obs, 3., backend=backend,
                            device=device).loss_and_cotangent(pred)[1])
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--validate', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--backend', default='pysdtw', choices=['pysdtw', 'reference'])
    p.add_argument('--traces', type=int, default=3)
    args = p.parse_args()
    report = validate(args.validate, device=args.device, backend=args.backend,
                      traces=args.traces)
    print(json.dumps({'passed': report['passed'], 'output': str(args.validate),
                      'stages': [{'cutoff':r['configuration']['stage_cutoff_hz'],
                                  'best_relative_error':min(s['relative_error'] for s in r['sweep']),
                                  'ad_dot':r['ad_dot'], 'seconds':r['vjp_seconds']}
                                 for r in report['stages']]}))
    raise SystemExit(0 if report['passed'] else 1)
