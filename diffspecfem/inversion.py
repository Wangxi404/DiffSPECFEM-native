"""Full-shot Marmousi-II inversion with native SPECFEM2D propagation."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from scripts.build_specfem import verify_build
from scripts.prepare_marmousi_inputs import prepare_inputs

from . import native
from .checkpoint import fingerprint_hash


ASSET_SHA256 = {
    'model_fields.npz': '17a59061b2c68095969fbc68ab2901ecc41ac8774cf1bfc70370bd1c8a424b93',
    'training.npz': '1e136e072fa4ea7b0d651916baee486358d8dab545c91a63d8cca67e4fefc7d7',
    'truth_for_observations_and_offline_only.npz': '3c647c65494a92ca6a58744b55da4e67ea2c1d5fa6410e7872e58d488a9156f0',
    'protocol.json': 'd6f094eeb1068fdb6ce0f7291167391160e13d1ca5c0b20ed396881ea1441579',
}
_WORKER_SOLVER = None
_WORKER_TRAINING = None


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def allowed_cpus(workers):
    if workers < 1:
        raise ValueError('--workers must be positive')
    available = (sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                 else list(range(os.cpu_count() or 1)))
    return available[:min(workers, len(available))]


def validate_config(config):
    """Reject settings that the fixed native/material/data maps do not implement."""
    fixed = {
        'native': dict(nx=50, nz=17, dt=.001, nt=7500, nrec=200, nproc=1,
                       arithmetic='float64'),
        'acquisition': dict(shots=40, source_x_indices=list(range(2, 198, 5)),
                            receiver_x_indices=list(range(200)), depth_index=2,
                            observation_components=['Ux', 'Uz'], observation_nt=2500,
                            observation_dt_s=.003),
        'objective': dict(backend='pysdtw', dtype='float64', gamma=.1,
                          processed_nt=313, decimation=8, anti_alias_cutoff_hz=10.,
                          anti_alias_transition_hz=2., fixed_observed_rms=True,
                          prediction_self_both_arguments=True),
    }
    for section, values in fixed.items():
        for name, expected in values.items():
            if config[section].get(name) != expected:
                raise ValueError(f'{section}.{name} is not implemented by this fixed example; '
                                 'adapt and revalidate the native/data maps for a separate case')


def validate_inputs(asset_dir):
    """Verify the prepared material fields and load the optimizer inputs."""
    asset_dir = Path(asset_dir)
    for name, expected in ASSET_SHA256.items():
        if file_hash(asset_dir/name) != expected:
            raise ValueError('Prepared example input changed: '+name)
    with np.load(asset_dir/'training.npz', allow_pickle=False) as archive:
        training = {key: archive[key].copy() for key in archive.files}
    if {'vp_true', 'vp', 'target_vp'} & set(training):
        raise ValueError('Unknown true Vp must not be an optimizer input')
    for key in ('vp_initial', 'vs_fixed', 'rho_fixed'):
        if training[key].shape != (68, 200) or not np.isfinite(training[key]).all():
            raise ValueError('Invalid fixed example field: '+key)
    return training


def _worker_init(source_dir, binary_dir, training_path, cpus, counter):
    global _WORKER_SOLVER, _WORKER_TRAINING
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    with counter.get_lock():
        slot = counter.value
        counter.value += 1
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {cpus[slot % len(cpus)]})
    _WORKER_SOLVER = native.NativeSolver(source_dir, binary_dir)
    with np.load(training_path, allow_pickle=False) as data:
        _WORKER_TRAINING = {key: data[key].copy() for key in data.files}


def _forward_shot(arguments):
    shot, path, model_path, record = arguments
    path = Path(path)
    vp = np.load(model_path, allow_pickle=False)
    _WORKER_SOLVER.set_model(path, vp, _WORKER_TRAINING['vs_fixed'],
                             _WORKER_TRAINING['rho_fixed'])
    data, timing = _WORKER_SOLVER.forward(path, record=record)
    return dict(shot=shot, model_sha256=native.digest(vp),
                native_model_sha256=file_hash(path/'DATA/proc000000_vp.bin'),
                data=native.sample(data), timing=timing,
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)


def _adjoint_shot(arguments):
    shot, path = arguments
    path = Path(path)
    cotangent = np.load(path/'cotangent_native.npy', allow_pickle=False)
    gradient, timing, _ = _WORKER_SOLVER.adjoint(path, cotangent, binary=True)
    return dict(shot=shot, gradient=gradient, timing=timing)


def stage_for_update(index, config):
    end = 0
    for item in config['inversion']['stages']:
        end += item['updates']
        if index < end:
            return item['cutoff_hz']
    raise ValueError('Update index lies outside the configured stage schedule')


def optimizer_step(initial, vp, theta, moment, second, mean_gradient, taper,
                   fixed_vs, index, config):
    """Take one Adam step in tapered, scaled Vp coordinates."""
    settings = config['inversion']
    scale = settings['velocity_scale_m_s']
    low = np.sqrt(4./3.*fixed_vs.astype(np.float64)**2+settings['bulk_speed_floor_m_s']**2)
    high = settings['upper_vp_m_s']
    theta_gradient = scale*taper*mean_gradient
    maximum = float(np.max(np.abs(theta_gradient)))
    if not np.isfinite(maximum) or maximum <= 0:
        raise FloatingPointError('Full-shot model gradient is nonfinite or zero')
    normalized = theta_gradient/maximum
    beta1, beta2 = settings['adam_betas']
    # Form the complements of the decimal coefficients from the configuration.
    weight1, weight2 = (float(1 - Decimal(str(beta))) for beta in (beta1, beta2))
    moment = beta1*moment+weight1*normalized
    second = beta2*second+weight2*normalized**2
    projected_candidate = initial+scale*taper*(theta-theta_gradient)
    projected_vp = np.minimum(np.maximum(projected_candidate, low), high)
    projected_vp[taper == 0] = initial[taper == 0]
    projected_theta = np.zeros_like(theta)
    np.divide(projected_vp-initial, scale*taper, out=projected_theta, where=taper > 0)
    projected_norm = float(np.max(np.abs(theta-projected_theta)))
    mhat = moment/(1.-beta1**(index+1))
    vhat = second/(1.-beta2**(index+1))
    direction = mhat/(np.sqrt(vhat)+settings['adam_epsilon'])
    rate = settings['learning_rate_m_s']*(settings['step_lr_gamma']**
                                           (index//settings['step_lr_step_size']))
    candidate_theta = theta-(rate/scale)*direction
    candidate = initial+scale*taper*candidate_theta
    updated = np.minimum(np.maximum(candidate, low), high)
    updated[taper == 0] = initial[taper == 0]
    updated_theta = np.zeros_like(theta)
    np.divide(updated-initial, scale*taper, out=updated_theta, where=taper > 0)
    if not np.isfinite(updated).all():
        raise FloatingPointError('Model update is nonfinite')
    return dict(vp=updated, theta=updated_theta, moment=moment, second_moment=second,
                theta_gradient=theta_gradient, normalized_gradient=normalized,
                adam_direction=direction, update=updated-vp,
                projected_gradient_norm=projected_norm, learning_rate_m_s=rate)


class Inversion:
    def __init__(self, project_root, output, config, *, workers=8, device='cpu',
                 source_dir=None, binary_dir=None):
        self.root = Path(project_root).resolve()
        self.output = Path(output).resolve()
        validate_config(config)
        self.config = config
        self.source = Path(source_dir or self.root/'build/specfem2d').resolve()
        self.binary = Path(binary_dir or self.source/'bin').resolve()
        self.solver = native.NativeSolver(self.source, self.binary)
        self.cpus = allowed_cpus(workers)
        self.device = device
        self.shots = list(range(config['acquisition']['shots']))
        if self.shots != list(range(40)):
            raise ValueError('This example requires all 40 independent shots')
        self.objectives = {}
        self.pool = None
        self.completed_updates = 0
        self.started = time.perf_counter()
        self.active = False

    def shot_path(self, shot):
        return self.output/'native-shots'/f'shot{shot:02d}'

    def status(self, phase, **details):
        write_json(self.output/'status.json', dict(phase=phase, pid=os.getpid(),
                   completed_updates=self.completed_updates, updated_unix=time.time(), **details))
        print(f'{phase}: '+json.dumps(details), flush=True)

    def ledger(self, event, **details):
        append_json(self.output/'ledger.jsonl', dict(event=event, **details))

    def prepare(self, resume=False):
        nonempty = self.output.exists() and any(self.output.iterdir())
        if nonempty and not resume:
            raise FileExistsError('Output is not empty; use --resume explicitly or choose --output')
        if resume and not nonempty:
            raise FileNotFoundError('--resume requires an existing run directory')
        build_manifest = verify_build(self.source, self.binary)
        fingerprint = dict(configuration=self.config, inputs=ASSET_SHA256,
                           native_binary_sha256=build_manifest['binary_sha256'],
                           interface_sha256={name: file_hash(Path(__file__).parent/name)
                                             for name in ('native.py', 'objective.py', 'soft_dtw_objective.py', 'inversion.py')})
        current_hash = fingerprint_hash(fingerprint)
        previous = self.output/'run.json'
        if resume:
            if not previous.is_file() or json.loads(previous.read_text()).get('fingerprint_sha256') != current_hash:
                raise ValueError('Run inputs, code or native binaries differ; choose a new output directory')
        self.assets = prepare_inputs(self.root/'assets/marmousi2d')
        self.training = validate_inputs(self.assets)
        if not resume:
            self.output.mkdir(parents=True, exist_ok=True)
            write_json(previous, dict(fingerprint_sha256=current_hash, **fingerprint))
        (self.output/'checkpoints').mkdir(exist_ok=True)
        self.active = True
        write_json(self.output/'execution.json', dict(workers=len(self.cpus), allowed_worker_cpus=self.cpus,
                   objective_device=self.device, inner_threads=1,
                   native_execution='independent singleton MPI processes; CPU Fortran',
                   multiprocessing_start_method='spawn', requested_resume=resume))
        template = self.output/'native-template'
        if not (template/'DATA/proc000000_layout.bin').exists():
            self.status('preparing_native_layout')
            self.solver.setup_template(template)
        for shot in self.shots:
            path = self.shot_path(shot)
            if (path/'case.json').exists():
                continue
            path.mkdir(parents=True)
            shutil.copytree(template/'DATA', path/'DATA')
            (path/'OUTPUT_FILES').mkdir()
            (path/'SEM').mkdir()
            case = json.loads((template/'case.json').read_text())
            case['shot'] = shot
            write_json(path/'case.json', case)
            native.edit_par(path/'DATA/SOURCE', xs=(2+5*shot)*45.)
        context = multiprocessing.get_context('spawn')
        counter = context.Value('i', 0)
        self.pool = ProcessPoolExecutor(max_workers=len(self.cpus), mp_context=context,
                initializer=_worker_init, initargs=(str(self.source), str(self.binary),
                str(self.assets/'training.npz'), self.cpus, counter))

    def preflight(self):
        """Check the native map/sampling transposes and actual data-objective VJP."""
        import torch
        from .objective import validate
        from .soft_dtw_objective import backend_provenance
        torch.set_num_threads(1)
        selected = torch.device(self.device)
        if selected.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but PyTorch cannot use it; choose --device cpu')
        backend = backend_provenance()  # fail early if the production backend is unavailable
        rng = np.random.default_rng(61)
        x, z, _, _ = self.solver.layout(self.output/'native-template')
        mapping = native.mapping(x, z)
        model_direction = rng.normal(size=(68, 200))
        native_direction = rng.normal(size=len(x))
        left = float(np.sum(native.interp(model_direction, mapping)*native_direction))
        right = float(np.sum(model_direction*native.interp_T(native_direction, mapping)))
        model_error = abs(left-right)/max(abs(left), abs(right), 1e-30)
        full = rng.normal(size=(2, 2, 7500))
        sampled = rng.normal(size=(2, 2, 2500))
        left = float(np.sum(native.sample(full)*sampled))
        right = float(np.sum(full*native.sample_T(sampled)))
        sampling_error = abs(left-right)/max(abs(left), abs(right), 1e-30)
        if max(model_error, sampling_error) >= 1e-12:
            raise ValueError('Model map or sampling transpose preflight failed')
        self.status('checking_data_objective_vjp')
        objective = validate(self.output/'data_objective_check.json', device=self.device,
                             backend='pysdtw', traces=2)
        if not objective['passed']:
            raise ValueError('Actual objective data VJP preflight failed')
        write_json(self.output/'preflight.json', dict(status='passed', model_map_dot_relative=model_error,
                   sampling_dot_relative=sampling_error, actual_data_objective_vjp='passed',
                   backend=backend, native_model_derivative='not checked by these data-only controls'))

    def generate_observations(self):
        directory = self.output/'observations'
        manifest_path = directory/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('input_model_sha256') != ASSET_SHA256['model_fields.npz']:
                raise ValueError('Saved observations belong to a different input model')
            if manifest['shot_ids'] != self.shots or any(file_hash(directory/name) != digest
                    for name, digest in manifest['files_sha256'].items()):
                raise ValueError('Saved observation set changed or is incomplete')
            return
        directory.mkdir(exist_ok=True)
        with np.load(self.assets/'truth_for_observations_and_offline_only.npz', allow_pickle=False) as data:
            true_vp = data['vp_true'].astype(np.float64)
        model_path = directory/'observation_model.npy'
        np.save(model_path, true_vp)
        self.status('generating_native_observations', shots=len(self.shots))
        futures = [self.pool.submit(_forward_shot, (shot, str(self.shot_path(shot)), str(model_path), False))
                   for shot in self.shots]
        hashes = {}
        for future in as_completed(futures):
            result = future.result()
            target = directory/f'shot{result["shot"]:02d}.npy'
            np.save(target, result['data'])
            hashes[target.name] = file_hash(target)
        model_path.unlink()
        write_json(manifest_path, dict(shot_ids=self.shots, files_sha256=hashes,
                   input_model_sha256=ASSET_SHA256['model_fields.npz'],
                   truth_model_sha256=native.digest(true_vp), observable='native Ux/Uz displacement (m)',
                   observation_dt_s=.003, observation_nt=2500, time_origin_s=-.24,
                   action='Separate native observation generation; no optimizer exists in this action'))

    def objective(self, shot, stage):
        from .objective import Fast40Objective
        key = (shot, stage)
        if key not in self.objectives:
            self.objectives = {k: v for k, v in self.objectives.items() if k[1] == stage}
            observed = np.load(self.output/'observations'/f'shot{shot:02d}.npy', allow_pickle=False)
            self.objectives[key] = Fast40Objective(observed, stage=stage, backend='pysdtw', device=self.device)
        return self.objectives[key]

    def evaluate(self, vp, stage, identifier, *, gradient=True, shots=None):
        shots = self.shots if shots is None else list(shots)
        start = time.perf_counter()
        model_hash = native.digest(vp)
        model_path = self.output/(identifier+'_model.npy')
        np.save(model_path, vp)
        self.ledger('evaluation_start', evaluation_id=identifier, shot_ids=shots,
                    model_sha256=model_hash, stage=stage, reduction='mean')
        self.status('native_forward_and_data_ad', evaluation_id=identifier, stage=stage, shots=len(shots))
        futures = [self.pool.submit(_forward_shot, (shot, str(self.shot_path(shot)), str(model_path), gradient))
                   for shot in shots]
        losses, native_hashes, data_ad_seconds = {}, set(), 0.
        forward_details = []
        for future in as_completed(futures):
            result = future.result()
            shot = result['shot']
            if result['model_sha256'] != model_hash:
                raise ValueError('Shots did not use one common current model')
            native_hashes.add(result['native_model_sha256'])
            ad_start = time.perf_counter()
            objective = self.objective(shot, stage)
            if gradient:
                loss, cotangent = objective.loss_and_cotangent(result.pop('data'))
                np.save(self.shot_path(shot)/'cotangent_native.npy', native.sample_T(cotangent))
            else:
                loss = objective.value(result.pop('data'))
            data_ad_seconds += time.perf_counter()-ad_start
            losses[shot] = loss
            forward_details.append(result)
            self.ledger('shot_batch', evaluation_id=identifier, shot_ids=[shot],
                        model_sha256=model_hash, native_model_sha256=result['native_model_sha256'])
        if set(losses) != set(shots) or len(native_hashes) != 1:
            raise ValueError('Shot reduction is incomplete or native material fields differ')
        forward_and_data_wall = time.perf_counter()-start
        gradient_sum = None
        adjoint_start = time.perf_counter()
        if gradient:
            self.status('native_adjoint', evaluation_id=identifier, shots=len(shots))
            tasks = {shot: self.pool.submit(_adjoint_shot, (shot, str(self.shot_path(shot)))) for shot in shots}
            # Sum in shot-ID order, independent of task completion order.
            gradient_sum = np.zeros_like(vp, dtype=np.float64)
            for shot in shots:
                result = tasks[shot].result()
                if result['shot'] != shot:
                    raise ValueError('Adjoint shot identity mismatch')
                gradient_sum += result['gradient']
        self.ledger('evaluation_complete', evaluation_id=identifier, model_sha256=model_hash,
                    shot_ids=shots, normalizer=len(shots), all_shots_reduced=True)
        model_path.unlink()
        return dict(loss=float(sum(losses[shot] for shot in shots)/len(shots)),
                    shot_losses=[losses[shot] for shot in shots], model_sha256=model_hash,
                    raw_sum_gradient=gradient_sum,
                    mean_gradient=None if gradient_sum is None else gradient_sum/len(shots),
                    full_gradient_wall_s=time.perf_counter()-start,
                    forward_and_data_ad_wall_s=forward_and_data_wall,
                    data_ad_and_sampling_work_s=data_ad_seconds,
                    adjoint_wall_s=time.perf_counter()-adjoint_start)

    def check_native_gradient(self):
        """Optional single-shot model directional check on the actual native path."""
        initial = self.training['vp_initial'].astype(np.float64)
        base = self.evaluate(initial, 3., 'native-check-base', shots=[0])
        z, x = np.indices(initial.shape)
        direction = native.taper()*(.7*np.cos(x/22.)+np.sin(z/12.)+.3)
        direction /= np.max(np.abs(direction))
        dot = float(np.sum(base['mean_gradient']*direction))
        rows = []
        for step in (2., 1.):
            plus = self.evaluate(initial+step*direction, 3., f'native-check-plus-{step}', gradient=False, shots=[0])['loss']
            minus = self.evaluate(initial-step*direction, 3., f'native-check-minus-{step}', gradient=False, shots=[0])['loss']
            difference = (plus-minus)/(2*step)
            rows.append(dict(step_m_s=step, adjoint_dot=dot, centered_difference=difference,
                             relative_error=abs(difference-dot)/max(abs(difference), abs(dot), 1e-30)))
        passed = abs(dot) > 1e-12 and min(row['relative_error'] for row in rows) < .005
        write_json(self.output/'native_gradient_check.json', dict(status='passed' if passed else 'failed',
                   shots=[0], stages=[3.], steps=rows, scope='Single-shot native directional check, not all-physics validation'))
        if not passed:
            raise ValueError('Native single-shot model directional check failed')

    def invert(self, maximum_updates=300, resume=False):
        initial = self.training['vp_initial'].astype(np.float64)
        taper = native.taper()
        vp = initial.copy()
        theta = np.zeros_like(initial)
        moment, second = theta.copy(), theta.copy()
        metrics = []
        metrics_path = self.output/'metrics.jsonl'
        if metrics_path.exists():
            metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        checkpoints = sorted((self.output/'checkpoints').glob('iter*.npz'))
        if checkpoints:
            if not resume:
                raise ValueError('Existing checkpoints require --resume')
            with np.load(checkpoints[-1], allow_pickle=False) as state:
                if str(state.get('input_model_sha256', '')) != ASSET_SHA256['model_fields.npz']:
                    raise ValueError('Checkpoint belongs to a different input model')
                vp, theta = state['vp'].copy(), state['theta'].copy()
                moment, second = state['moment'].copy(), state['second_moment'].copy()
                self.completed_updates = int(state['iteration'])
                if native.digest(vp) != str(state['model_after_sha256']):
                    raise ValueError('Checkpoint model hash mismatch')
            if not metrics or metrics[-1]['iteration'] != self.completed_updates or metrics[-1]['model_after_sha256'] != native.digest(vp):
                raise ValueError('Checkpoint and iteration metrics disagree; retain and inspect the interrupted commit')
        if maximum_updates < self.completed_updates:
            raise ValueError('--max-updates is below the already accepted update count')
        reason = f'budget_exhaustion_{maximum_updates}_updates'
        for index in range(self.completed_updates, maximum_updates):
            stage = stage_for_update(index, self.config)
            before_hash = native.digest(vp)
            identifier = f'update{index+1:04d}'
            iteration_start = time.perf_counter()
            self.ledger('iteration_start', iteration_id=identifier, shot_ids=self.shots,
                        model_sha256=before_hash, reduction='mean', stage=stage)
            result = self.evaluate(vp, stage, identifier)
            step = optimizer_step(initial, vp, theta, moment, second, result['mean_gradient'],
                                  taper, self.training['vs_fixed'], index, self.config)
            vp, theta = step['vp'], step['theta']
            moment, second = step['moment'], step['second_moment']
            initial_norm = metrics[0]['projected_gradient_norm'] if metrics else step['projected_gradient_norm']
            relative_gradient = step['projected_gradient_norm']/max(initial_norm, 1e-30)
            metric = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
            metric.update(iteration=index+1, stage=stage, learning_rate_m_s=step['learning_rate_m_s'],
                          max_physical_step_m_s=float(np.max(np.abs(step['update']))),
                          projected_gradient_norm=step['projected_gradient_norm'],
                          relative_projected_gradient=relative_gradient,
                          model_after_sha256=native.digest(vp),
                          iteration_wall_s=time.perf_counter()-iteration_start,
                          loss_model='current model before the accepted update')
            save_npz(self.output/'checkpoints'/f'iter{index+1:04d}.npz', iteration=index+1,
                     vp=vp, theta=theta, moment=moment, second_moment=second,
                     input_model_sha256=ASSET_SHA256['model_fields.npz'],
                     raw_sum_gradient=result['raw_sum_gradient'], mean_gradient=result['mean_gradient'],
                     theta_gradient=step['theta_gradient'], normalized_gradient=step['normalized_gradient'],
                     adam_direction=step['adam_direction'], update=step['update'], loss=result['loss'],
                     stage=str(stage), model_before_sha256=before_hash, model_after_sha256=native.digest(vp))
            self.ledger('accepted_update', iteration_id=identifier, model_before_sha256=before_hash,
                        model_after_sha256=native.digest(vp), gradient_evaluation_id=identifier)
            self.ledger('iteration_end', iteration_id=identifier, status='accepted')
            append_json(metrics_path, metric)
            metrics.append(metric)
            self.completed_updates = index+1
            save_npz(self.output/'final_model.npz', vp=vp, initial=initial, iteration=index+1,
                     input_model_sha256=ASSET_SHA256['model_fields.npz'], stop_reason='in_progress', parameter_taper=taper)
            self.status('inversion_running', loss=result['loss'], stage=stage,
                        max_physical_step_m_s=metric['max_physical_step_m_s'])
            stopping = self.config['inversion']['stopping']
            recent = metrics[-stopping['window']:]
            if len(recent) == stopping['window'] and all(row['stage'] == stage for row in recent):
                relative_loss = abs(recent[-1]['loss']-recent[0]['loss'])/max(abs(recent[0]['loss']), 1e-30)
                if (relative_loss < stopping['relative_objective_tolerance'] and
                        metric['max_physical_step_m_s'] < stopping['maximum_physical_step_m_s'] and
                        relative_gradient < stopping['relative_projected_gradient_tolerance']):
                    reason = 'data_based_convergence'
                    break
        save_npz(self.output/'final_model.npz', vp=vp, initial=initial, iteration=self.completed_updates,
                 input_model_sha256=ASSET_SHA256['model_fields.npz'], stop_reason=reason, parameter_taper=taper)
        final = self.evaluate(vp, None, 'final-accepted-evaluation', gradient=False)
        write_json(self.output/'result.json', dict(completed_updates=self.completed_updates,
                   stop_reason=reason, maximum_updates=maximum_updates,
                   final_full_band_objective=final['loss'], final_model_sha256=native.digest(vp),
                   wall_s=time.perf_counter()-self.started,
                   objective=self.config['objective'].get('kind'), native_precision='float64',
                   physics='planar elastic P-SV', native_model_gradient_check='native_gradient_check.json if requested'))
        self.status('inversion_complete', stop_reason=reason, final_full_band_objective=final['loss'])

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None


def run_example(project_root, output, config, *, workers=8, device='cpu',
                maximum_updates=300, resume=False, check_only=False,
                check_gradient=False, source_dir=None, binary_dir=None):
    if not 1 <= maximum_updates <= config['inversion']['maximum_updates']:
        raise ValueError('--max-updates must be between 1 and the example budget of 300')
    runner = Inversion(project_root, output, config, workers=workers, device=device,
                       source_dir=source_dir, binary_dir=binary_dir)
    try:
        runner.prepare(resume=resume)
        runner.preflight()
        if not check_only or check_gradient:
            runner.generate_observations()
        if check_gradient:
            runner.check_native_gradient()
        if check_only:
            runner.status('preflight_complete', native_model_gradient='passed' if check_gradient else 'not requested')
            return
        runner.invert(maximum_updates=maximum_updates, resume=resume)
    except BaseException as error:
        if runner.active:
            runner.status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          error=repr(error), stop_reason='interruption' if isinstance(error, KeyboardInterrupt) else 'execution_failure')
            model = runner.output/'final_model.npz'
            if model.exists():
                with np.load(model, allow_pickle=False) as data:
                    saved = {key: data[key].copy() for key in data.files}
                saved['stop_reason'] = np.asarray('interruption' if isinstance(error, KeyboardInterrupt) else 'execution_failure')
                save_npz(model, **saved)
        raise
    finally:
        runner.close()
        if runner.active and not check_only and (runner.output/'final_model.npz').exists():
            from .plotting import render_run
            render_run(runner.root, runner.output)
