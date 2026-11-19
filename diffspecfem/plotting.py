"""Plot saved Vp models and loss histories."""
from pathlib import Path
import hashlib
import html
import json
import os

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scripts.prepare_marmousi_inputs import prepare_inputs


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _save(fig, path, credit=None):
    for extension in ('png', 'pdf', 'svg'):
        metadata = {('Subject' if extension == 'pdf' else 'Description'): credit} if credit else None
        fig.savefig(path.with_suffix('.'+extension), dpi=600, metadata=metadata)
    plt.close(fig)


def render_run(project_root, run_dir, output_dir=None):
    """Write model and loss figures and an HTML report."""
    project_root, run_dir = Path(project_root).resolve(), Path(run_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else run_dir
    figures = output/'figures'
    final = run_dir/'final_model.npz'
    if not final.is_file():
        raise FileNotFoundError(final)
    inputs = prepare_inputs(project_root/'assets/marmousi2d')/'model_fields.npz'
    result_file, metrics_file = run_dir/'result.json', run_dir/'metrics.jsonl'
    model_hash = file_hash(inputs)
    final_hash = file_hash(final)
    reference_objective = None
    with np.load(inputs, allow_pickle=False) as data:
        truth, initial = data['vp_true'].copy(), data['vp_initial'].copy()
        x, z = data['x_m'].copy(), data['depth_m'].copy()
    with np.load(final, allow_pickle=False) as data:
        source_hash = str(data.get('input_model_sha256', ''))
        if not source_hash:
            paper = json.loads((project_root/'docs/source-provenance.json').read_bytes())['reference_result']
            if final_hash == paper['checkpoint_file_sha256']:
                source_hash = paper['input_source']['model_sha256']
                reference_objective = paper['objective']
        if source_hash != model_hash or not np.array_equal(data['initial'], initial):
            raise ValueError('Checkpoint and displayed model inputs come from different datasets')
        inverted = data['vp'].copy()
        iteration = int(data['iteration'])
        stop = str(data['stop_reason'])
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    run_file = run_dir/'run.json'
    run = json.loads(run_file.read_text()) if run_file.is_file() else {}
    objective = (run.get('configuration', {}).get('objective', {}).get('kind')
                 or result.get('objective') or reference_objective)
    if objective == 'soft_dtw_divergence':
        objective = 'debiased soft-DTW divergence'
    objective_label = ' · ' + html.escape(str(objective)) if objective else ''
    result_count = int(result.get('completed_updates', iteration))
    if result_count > iteration:
        raise ValueError('Result claims updates beyond the accepted checkpoint')
    # An interruption can leave result.json behind the latest checkpoint.
    result_metadata_current = (result_count == iteration and
                               result.get('stop_reason', stop) == stop)
    records = [json.loads(line) for line in metrics_file.read_text().splitlines() if line.strip()]
    records = [record for record in records if int(record['iteration']) <= iteration]
    if len(records) != iteration or [int(r['iteration']) for r in records] != list(range(1, iteration+1)):
        raise ValueError('Loss records must cover every accepted update exactly once')
    fields = [truth, initial, inverted]
    if any(a.shape != (68, 200) or not np.isfinite(a).all() for a in fields):
        raise ValueError('Expected full finite 68 x 200 saved models')
    losses = np.asarray([r['loss'] for r in records], dtype=float)
    if not np.isfinite(losses).all():
        raise ValueError('Loss records must be finite')
    figures.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.linewidth': .55, 'xtick.major.width': .5,
                         'ytick.major.width': .5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    displayed = [a.astype(np.float64)/1000. for a in fields]
    limits = [min(float(a.min()) for a in displayed), max(float(a.max()) for a in displayed)]
    dx, dz = np.diff(x)[0], np.diff(z)[0]
    extent = [(x[0]-dx/2)/1000., (x[-1]+dx/2)/1000.,
              (z[-1]+dz/2)/1000., (z[0]-dz/2)/1000.]
    fig = plt.figure(figsize=(10.5, 2.65))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, .035],
                            left=.055, right=.95, bottom=.22, top=.88, wspace=.09)
    axes = [fig.add_subplot(grid[0, k]) for k in range(3)]
    for i, (ax, values, title) in enumerate(zip(axes, displayed, ['True', 'Initial', 'Inverted'])):
        image = ax.imshow(values, cmap='Blues', vmin=limits[0], vmax=limits[1],
                          extent=extent, origin='upper', aspect='equal', interpolation='nearest')
        ax.set_title(title, pad=8)
        ax.set_xticks([6, 9, 12])
        ax.set_yticks([1, 2, 3])
        if i:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel('Depth (km)')
    axes[1].set_xlabel('Distance (km)')
    fig.canvas.draw()
    position = axes[-1].get_position()
    slot = grid[0, 3].get_position(fig)
    color_axes = fig.add_axes([slot.x0, position.y0, slot.width, position.height])
    bar = fig.colorbar(image, cax=color_axes)
    bar.set_label('Vp (km/s)', fontsize=9)
    bar.outline.set_linewidth(.55)
    _save(fig, figures/'models', credit=(
        'University of Houston AGL download edition of Marmousi-II by Martin, Wiley and Marfurt. '
        'Model Definition files: https://www.agl.uh.edu/downloads/downloads.htm. '
        'Cropped and sampled; initial Vp was Gaussian-smoothed.'))

    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.subplots_adjust(left=.12, right=.98, bottom=.24, top=.82)
    cuts = [0]+[i for i in range(1, len(records)) if records[i]['stage'] != records[i-1]['stage']]+[len(records)]
    for start, end in zip(cuts[:-1], cuts[1:]):
        points = records[start:end]
        ax.plot([r['iteration'] for r in points], [r['loss'] for r in points],
                color='#2166ac', linewidth=1.25,
                marker='o' if len(points) == 1 else None, markersize=3)
        if start:
            ax.axvline(points[0]['iteration']-.5, color='#808080', linewidth=.6, linestyle='--')
        stage = points[0]['stage']
        label = 'Full band' if stage is None else f'{stage:g} Hz'
        midpoint = (points[0]['iteration']+points[-1]['iteration'])/2
        ax.text(midpoint, 1.06, label, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8)
    if len(losses) and (losses > 0).all():
        ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e7ecf0', linewidth=.45)
    _save(fig, figures/'loss')

    status = 'Budget limit reached' if 'budget' in stop else stop.replace('_', ' ').capitalize()
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DiffSPECFEM-native — Marmousi-II</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#172c40;background:#f6f8fa;margin:0}}
main{{max-width:1150px;margin:40px auto;padding:28px;background:white;border-radius:12px}}
h1{{font-size:26px;margin:0}}p{{color:#526477}}img{{display:block;width:100%;height:auto}}
h2{{font-size:18px;margin-top:30px}}a{{color:#2166ac}}</style><main>
<h1>DiffSPECFEM-native · Marmousi-II inversion</h1>
<p>P-SV · native float64{objective_label}<br>
{iteration} accepted updates · {html.escape(status)}</p>
<img src="figures/models.png" alt="True, Initial and Inverted Vp with one shared Blues scale">
<p>Marmousi-II by Martin, Wiley and Marfurt, © 2004 Allied Geophysical Laboratory, University of Houston.
We cropped and sampled the University of Houston AGL download edition and smoothed Vp for the initial model.
AGL lists this edition as Level 3 – Limited.
<a href="https://www.agl.uh.edu/downloads/downloads.htm">AGL model</a> ·
<a href="https://www.agl.uh.edu/resources-data.php">AGL terms</a> ·
<a href="https://doi.org/10.1190/1.2172306">Model paper</a>.</p>
<h2>Loss</h2><img src="figures/loss.png" alt="Loss with separate frequency stages">
<p>Loss is evaluated before each update and plotted separately for each frequency stage.</p>
<p><a href="figures/models.pdf">Models PDF</a> · <a href="figures/loss.pdf">Loss PDF</a></p>
</main></html>'''
    report = output/'model-recovery.html'
    report.write_text(page)
    manifest = dict(accepted_updates=iteration, stop_reason=stop, colormap='Blues',
        shared_vp_limits_km_s=limits, model_shape_depth_x=[68, 200],
        input_edition='UH AGL download edition', input_npz_sha256=model_hash, final_npz_sha256=final_hash,
        metrics_sha256=file_hash(metrics_file), all_original_samples_retained=True,
        result_metadata_matches_checkpoint=result_metadata_current,
        interpolation='nearest; no display smoothing', loss_records=len(records),
        array_sha256={key: array_hash(value) for key, value in zip(
            ['vp_true', 'vp_initial', 'vp_inverted'], fields)})
    (figures/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return report
