"""Publish the three paper criteria from existing observed rubric summaries."""
from __future__ import annotations

import csv

CRITERIA = {
    'task_completion': 'Task completion',
    'data_retrieval_accuracy': 'Data retrieval accuracy',
    'generalized_result_verification': 'Result verification',
}


def criterion_rows(groups, names, *, repeated=False):
    rows = []
    for key, name in names.items():
        group = groups[key]
        cases = group['pooled_cases'] if repeated else group['cases']
        metrics = {}
        for criterion in CRITERIA:
            observed = cases['rubric_success_rates'].get(criterion, {})
            average = group['means']['rubric_success_rates'].get(criterion, {}) if repeated else {}
            metrics[criterion] = {
                'mean': average.get('mean') if repeated else observed.get('success_rate'),
                'sd': average.get('sd') if repeated else None,
                'observed_judgments': observed.get('observed', 0),
                'observed_repetitions': average.get('observed_repetitions', 0) if repeated else int(bool(observed)),
            }
        rows.append({'model': name, 'target': key, 'judged_trials': cases['graded'],
                     'assigned_trials': cases.get('assigned_cases', cases['attempted']), 'metrics': metrics})
    return rows


def publish_criterion_averages(dest, rows, colors, *, repeated=False):
    """Render averages without changing the underlying scores or pass decisions."""
    fields = ['model', 'target', 'criterion', 'mean', 'sd', 'observed_judgments',
              'observed_repetitions', 'judged_trials', 'assigned_trials']
    with (dest / 'criterion-averages.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            for criterion, metric in row['metrics'].items():
                writer.writerow({k: v for k, v in row.items() if k != 'metrics'} |
                                {'criterion': criterion} | metric)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), layout='constrained')
    for ax, (criterion, title) in zip(axes, CRITERIA.items()):
        for index, (row, color) in enumerate(zip(rows, colors)):
            metric = row['metrics'][criterion]
            if metric['mean'] is None:
                continue
            value = metric['mean'] * 100
            sd = metric['sd'] * 100 if metric['sd'] is not None else None
            ax.barh(index, value, height=.55, color=color)
            if sd is not None:
                ax.errorbar(value, index, xerr=sd, fmt='none', ecolor='#24382a', capsize=3)
            # Keep annotations inside the plot even for averages near 100%.
            ax.text(2, index, f'{value:.1f}%', va='center', fontsize=9,
                    color='white' if index < 2 and value > 30 else '#24382a')
        ax.set_yticks(range(len(rows)), [row['model'] for row in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(title, loc='left', fontweight='bold', pad=12)
        ax.set_xlabel('Mean True judgments (%)' + (' ± sample SD across repetitions' if repeated else ''))
    fig.savefig(dest / 'graphs/criterion-averages.png', dpi=180, bbox_inches='tight')
    svg = dest / 'graphs/criterion-averages.svg'
    fig.savefig(svg, bbox_inches='tight', metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)

    table = ['| Model | Task completion (%) | Data retrieval accuracy (%) | Result verification (%) | Judged / assigned |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        cells = []
        for metric in row['metrics'].values():
            cell = '—' if metric['mean'] is None else f"{metric['mean'] * 100:.1f}"
            if metric['sd'] is not None:
                cell += f" ± {metric['sd'] * 100:.1f}"
            cells.append(cell)
        table.append(f"| {row['model']} | {' | '.join(cells)} | {row['judged_trials']}/{row['assigned_trials']} |")
    return '\n'.join(table)
