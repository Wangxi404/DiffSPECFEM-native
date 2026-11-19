"""File interface to native SPECFEM2D propagation and its native adjoint.

The numerical maps and source conventions retain the tested Marmousi-II
application. Python handles material/data I/O; no wave propagator is in AD.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import time

import numpy as np


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<f8').tobytes()).hexdigest()


def edit_par(path, **changes):
    path = Path(path)
    text = path.read_text()
    for name, value in changes.items():
        if isinstance(value, bool):
            value = '.true.' if value else '.false.'
        text, count = re.subn(r'(?m)^(\s*'+re.escape(name)+r'\s*=)[^\n]*',
                              lambda match: match[1]+' '+str(value), text)
        if count != 1:
            raise ValueError(f'{path}: expected exactly one {name}, found {count}')
    path.write_text(text)


def mapping(x, z):
    # Native z is upward; physical depth is 3515-native_z.
    a = np.clip(x/45., 0, 199.)
    b = np.clip((3015.-z)/45., 0, 67.)
    ix, iz = np.minimum(a.astype(int), 198), np.minimum(b.astype(int), 66)
    ax, az = a-ix, b-iz
    indices = np.stack((iz*200+ix, iz*200+ix+1,
                        (iz+1)*200+ix, (iz+1)*200+ix+1), axis=1)
    weights = np.stack(((1-ax)*(1-az), ax*(1-az), (1-ax)*az, ax*az), axis=1)
    return indices, weights


def interp(value, material_map):
    indices, weights = material_map
    return (np.asarray(value).ravel()[indices]*weights).sum(axis=1)


def interp_T(gradient, material_map):
    indices, weights = material_map
    return np.bincount(indices.ravel(), weights=(gradient[:, None]*weights).ravel(),
                       minlength=68*200).reshape(68, 200)


def write_record(path, values):
    values = np.ascontiguousarray(values, dtype='<f8')
    count = np.array(values.nbytes, dtype='<i4')
    with Path(path).open('wb') as stream:
        count.tofile(stream)
        values.tofile(stream)
        count.tofile(stream)


def sample(data, dt=.001):
    # C = sample selection after a symmetric, zero-padded FFT anti-alias filter.
    nt = data.shape[-1]
    if nt != 7500 or abs(dt-.001) >= 1e-15:
        raise ValueError('The tested sampling map requires 7500 native samples at dt=0.001 s')
    frequency = np.fft.rfftfreq(3*nt, dt)
    response = np.where(frequency <= 120., 1., np.where(frequency >= 160., 0.,
                        .5*(1+np.cos(np.pi*(frequency-120.)/40.))))
    padded = np.pad(data, [(0, 0)]*(data.ndim-1)+[(nt, nt)])
    output = np.fft.irfft(np.fft.rfft(padded, axis=-1)*response,
                          n=3*nt, axis=-1)[..., nt:2*nt]
    return output[..., ::3]


def sample_T(cotangent, dt=.001):
    if cotangent.shape[-1] != 2500 or abs(dt-.001) >= 1e-15:
        raise ValueError('The tested sampling transpose requires 2500 objective samples')
    nt = 7500
    data = np.zeros(cotangent.shape[:-1]+(nt,))
    data[..., ::3] = cotangent
    frequency = np.fft.rfftfreq(3*nt, dt)
    response = np.where(frequency <= 120., 1., np.where(frequency >= 160., 0.,
                        .5*(1+np.cos(np.pi*(frequency-120.)/40.))))
    padded = np.pad(data, [(0, 0)]*(data.ndim-1)+[(nt, nt)])
    return np.fft.irfft(np.fft.rfft(padded, axis=-1)*response,
                       n=3*nt, axis=-1)[..., nt:2*nt]


def taper():
    depth, x = np.indices((68, 200))
    distance = np.minimum.reduce([x, 199-x, depth, 67-depth])*45.
    return np.sin(.5*np.pi*np.minimum(distance/45., 1.))**2


class NativeSolver:
    """Pickleable paths to a pinned upstream example and native executables."""

    def __init__(self, source_dir, binary_dir=None):
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.binary_dir = (Path(binary_dir).expanduser().resolve() if binary_dir is not None
                           else self.source_dir/'bin')

    def run(self, path, exe, phase):
        path = Path(path).resolve()
        if exe not in {'xmeshfem2D', 'xspecfem2D'}:
            raise ValueError('Expected xmeshfem2D or xspecfem2D')
        if Path(phase).name != phase:
            raise ValueError('A log phase must be a plain filename stem')
        binary = self.binary_dir/exe
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise FileNotFoundError(f'Native executable missing: {binary}; run scripts/build_specfem.py first')
        environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        log = path/(phase+'.log')
        started = time.perf_counter()
        with log.open('w') as stream:
            result = subprocess.run([str(binary)], cwd=path, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter()-started
        (path/(phase+'.timing.json')).write_text(json.dumps(dict(
            wall_seconds=elapsed, returncode=result.returncode,
            executable=exe, inner_threads=1), indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'{exe} failed with exit {result.returncode}; see {log}')
        return elapsed

    def prepare(self, path, nx=50, nz=17, dt=.001, nt=7500, nrec=200, shot=0, nproc=1):
        if nproc != 1:
            raise ValueError('This example uses independent singleton-MPI shots; nproc must be 1')
        if not 0 <= shot < 40:
            raise ValueError('The Marmousi-II example has shot IDs 0 through 39')
        path = Path(path)
        for name in ('DATA', 'OUTPUT_FILES', 'SEM'):
            (path/name).mkdir(parents=True, exist_ok=True)
        example = self.source_dir/'EXAMPLES/Tromp2005_kernel/DATA'
        for name in ('Par_file', 'SOURCE'):
            source = example/name
            if not source.is_file():
                raise FileNotFoundError(f'Pinned SPECFEM2D example input missing: {source}')
            (path/'DATA'/name).write_text(source.read_text())
        par = path/'DATA/Par_file'
        edit_par(par, title='DiffSPECFEM Marmousi-II', NPROC=nproc, NSTEP=nt,
                 DT=f'{dt:.17g}', USER_T0=0., P_SV=True, AXISYM=False,
                 time_stepping_scheme=1, ATTENUATION_VISCOELASTIC=False,
                 ATTENUATION_VISCOACOUSTIC=False, ATTENUATION_PORO_FLUID_PART=False,
                 UNDO_ATTENUATION_AND_OR_PML=False, PML_BOUNDARY_CONDITIONS=False,
                 STACEY_ABSORBING_CONDITIONS=True,
                 MODEL='default', SAVE_MODEL='ascii', SIMULATION_TYPE=1, SAVE_FORWARD=False,
                 NO_BACKWARD_RECONSTRUCTION=False, NTSTEP_BETWEEN_OUTPUT_SEISMOS=nt,
                 NTSTEP_BETWEEN_OUTPUT_SAMPLE=1, save_ASCII_seismograms=False,
                 save_binary_seismograms_double=True, use_existing_STATIONS=True,
                 output_color_image=False, output_postscript_snapshot=False,
                 output_wavefield_dumps=False, save_ASCII_kernels=False,
                 NTSTEP_BETWEEN_OUTPUT_INFO=nt, interfacesfile='interfaces.dat',
                 xmin=0., xmax=8955., nx=nx, APPROXIMATE_HESS_KL=False,
                 NUMBER_OF_SIMULTANEOUS_RUNS=1)
        text, material_count = re.subn(r'(?m)^1 1 2600[^\n]*',
                  '1 1 2500. 6000. 1500. 0 0 9999. 9999. 0 0 0 0 0 0', par.read_text())
        text, region_count = re.subn(r'(?m)^1 80\s+1 32 1', f'1 {nx} 1 {nz} 1', text)
        if material_count != 1 or region_count != 1:
            raise ValueError('Pinned SPECFEM2D material/region template does not match')
        par.write_text(text)
        (path/'DATA/interfaces.dat').write_text(
            f'2\n2\n0 0\n8955 0\n2\n0 3015\n8955 3015\n{nz}\n')
        edit_par(path/'DATA/SOURCE', xs=(2+shot*5)*45., zs=2925.,
                 time_function_type=8, name_of_source_file='./DATA/wavelet.txt',
                 f0=5., factor=1e10, anglesource=0.)
        elapsed = np.arange(nt)*dt
        tau = elapsed-.24
        wavelet = tau*np.exp(-np.pi**2*25.*tau*tau)
        np.savetxt(path/'DATA/wavelet.txt', np.column_stack((elapsed, wavelet)), fmt='%.17e')
        receiver_x = np.linspace(0, 8955, nrec) if nrec != 200 else np.arange(200)*45.
        (path/'DATA/STATIONS').write_text(''.join(
            f'R{i:04d} AA {x:.17g} 2925. 0. 0.\n' for i, x in enumerate(receiver_x)))
        configuration = dict(nx=nx, nz=nz, dt=dt, nt=nt, nrec=nrec, shot=shot,
            nproc=nproc, native_origin_s=-.24, physics='planar elastic P-SV',
            integrator='Newmark', native_dtype='float64',
            source='vertical line force, integrated Ricker tau exp(-pi^2*f0^2*tau^2), f0=5Hz, elapsed delay0.24s, factor1e10',
            physical_extent_m=[5000., 13955., 500., 3515.],
            native_extent_m=[0., 8955., 0., 3015.])
        (path/'case.json').write_text(json.dumps(configuration, indent=2)+'\n')
        return path

    def setup_template(self, path):
        path = self.prepare(path)
        self.run(path, 'xmeshfem2D', 'mesh-layout')
        self.run(path, 'xspecfem2D', 'forward-layout')
        self.layout(path)
        return path

    def layout(self, path, rank=0):
        filename = Path(path)/f'DATA/proc{rank:06d}_layout.bin'
        with filename.open('rb') as stream:
            header = np.fromfile(stream, dtype='<i4', count=3)
            if len(header) != 3:
                raise ValueError(f'{filename}: incomplete layout header')
            nspec, nx, nz = (int(v) for v in header)
            values = np.fromfile(stream, dtype='<f8')
        if min(nspec, nx, nz) <= 0 or values.size != 3*nspec*nx*nz:
            raise ValueError(f'{filename}: native layout dimensions do not match its payload')
        x, z, weight = values.reshape(3, -1)
        return x, z, weight, (nspec, nz, nx)

    def set_model(self, path, vp, vs, rho):
        path = Path(path)
        configuration = json.loads((path/'case.json').read_text())
        for name, field in [('vp', vp), ('vs', vs), ('rho', rho)]:
            field = np.asarray(field)
            if field.shape != (68, 200) or not np.isfinite(field).all():
                raise ValueError(f'{name}: finite 68x200 material field required')
        maps = []
        for rank in range(configuration['nproc']):
            x, z, weight, shape = self.layout(path, rank)
            material_map = mapping(x, z)
            maps.append(material_map)
            for name, field in [('vp', vp), ('vs', vs), ('rho', rho)]:
                write_record(path/f'DATA/proc{rank:06d}_{name}.bin', interp(field, material_map))
        edit_par(path/'DATA/Par_file', MODEL='binary', SAVE_MODEL='default')
        return maps

    def forward(self, path, record=False, fullhistory=False, wavefields=False):
        path = Path(path)
        edit_par(path/'DATA/Par_file', SIMULATION_TYPE=1, SAVE_FORWARD=record,
                 NO_BACKWARD_RECONSTRUCTION=fullhistory, output_wavefield_dumps=wavefields,
                 NTSTEP_BETWEEN_OUTPUT_IMAGES=500, use_binary_for_wavefield_dumps=True)
        mesh = self.run(path, 'xmeshfem2D', 'mesh-forward')
        forward = self.run(path, 'xspecfem2D', 'forward')
        configuration = json.loads((path/'case.json').read_text())
        nt, nrec = configuration['nt'], configuration['nrec']
        data = np.stack([np.fromfile(path/f'OUTPUT_FILES/U{component}_file_double_d.bin',
                        dtype='<f8').reshape(nrec, nt) for component in ['x', 'z']], axis=1)
        if not np.isfinite(data).all():
            raise FloatingPointError('Native forward produced nonfinite displacement samples')
        return data, dict(mesh=mesh, forward=forward)

    def adjoint(self, path, cotangent, source_scale=None, binary=False):
        path = Path(path)
        configuration = json.loads((path/'case.json').read_text())
        dt, nt, nrec = configuration['dt'], configuration['nt'], configuration['nrec']
        if cotangent.shape != (nrec, 2, nt) or not np.isfinite(cotangent).all():
            raise ValueError('Native cotangent must be finite with shape (receivers,2,native_time)')
        # Native source injection reverses these forward-ordered samples itself.
        scale = 1/dt if source_scale is None else source_scale
        times = np.arange(nt)*dt-.24
        for i in range(nrec):
            for j, channel in enumerate(['BXX', 'BXZ']):
                filename = path/f'SEM/AA.R{i:04d}.{channel}.adj'
                if binary:
                    np.asarray(cotangent[i, j]*scale, dtype='<f8').tofile(str(filename)+'.bin')
                else:
                    Path(str(filename)+'.bin').unlink(missing_ok=True)
                    np.savetxt(filename, np.column_stack((times, cotangent[i, j]*scale)), fmt='%.17e')
        edit_par(path/'DATA/Par_file', SIMULATION_TYPE=3, SAVE_FORWARD=False,
                 output_wavefield_dumps=False)
        mesh = self.run(path, 'xmeshfem2D', 'mesh-adjoint')
        adjoint = self.run(path, 'xspecfem2D', 'adjoint')
        gradient, local_gradients = np.zeros((68, 200)), []
        for rank in range(configuration['nproc']):
            x, z, weight, shape = self.layout(path, rank)
            local = np.fromfile(path/f'OUTPUT_FILES/proc{rank:06d}_vp_gradient.bin',
                                dtype='<f8')[:len(x)]
            if len(local) != len(x):
                raise ValueError('Native Vp gradient payload is incomplete')
            local_gradients.append(local)
            gradient += interp_T(local, mapping(x, z))
        if not np.isfinite(gradient).all():
            raise FloatingPointError('Native adjoint produced a nonfinite model gradient')
        return gradient, dict(mesh_adjoint=mesh, adjoint=adjoint), local_gradients
