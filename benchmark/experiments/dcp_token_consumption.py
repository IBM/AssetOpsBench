"""
generate_cost_figures.py
Produces all execution cost figures for the NeurIPS evaluation paper.
Input:  trajectory_stats.csv
Output: figures/fig_cost_*.pdf and figures/fig_cost_*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    "font.family"       : "serif",
    "font.serif"        : ["Georgia", "Times New Roman", "DejaVu Serif"],
    "font.size"         : 13,        # was 11
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.linewidth"    : 0.8,
    "figure.dpi"        : 300,
    "savefig.dpi"       : 300,
    "savefig.facecolor" : "white",
    "savefig.bbox"      : "tight",
    "xtick.labelsize"   : 12,        # new
    "ytick.labelsize"   : 12,        # new
    "axes.labelsize"    : 13,        # new
    "axes.titlesize"    : 13,        # new
    "legend.fontsize"   : 11,        # new
})

AGENT_COLORS = {
    'IoT' : '#2563EB',
    'FMSA': '#7C3AED',
    'TSFM': '#059669',
    'WO'  : '#D97706',
    'E2E' : '#DC2626',
}
CLASS_COLORS  = {'Single-Agent': '#2563EB', 'Multi-Agent': '#DC2626'}
PHASE_COLORS  = {'P1': '#1D4ED8', 'P2': '#B45309'}
PHASE_LABELS  = {'P1': 'Phase 1\n(Development)', 'P2': 'Phase 2\n(Evaluation)'}

legend_agent = [mpatches.Patch(color=v, label=k, alpha=0.85)
                for k, v in AGENT_COLORS.items()]
legend_class  = [mpatches.Patch(color=v, label=k, alpha=0.85)
                 for k, v in CLASS_COLORS.items()]

def load_data(path='trajectory_stats.csv'):
    df = pd.read_csv(path)
    df['api_calls'] = df['api_calls'] / 2

    PHASE1_IDS = [5, 8, 106, 114, 203, 204, 400, 405, 424, 604, 607]

    df['phase'] = df['id'].apply(
        lambda x: 'P1' if x in PHASE1_IDS else 'P2')

    def agent(sid):
        if sid in [5, 7, 8, 11]: return 'IoT'
        if 100 <= sid <= 199:    return 'FMSA'
        if 200 <= sid <= 299:    return 'TSFM'
        if 400 <= sid <= 499:    return 'WO'
        if 600 <= sid <= 699:    return 'E2E'
        return 'Unknown'

    df['agent']       = df['id'].map(agent)
    df['agent_class'] = df['agent'].map(
        lambda a: 'Multi-Agent' if a == 'E2E' else 'Single-Agent')
    return df


def savefig(name):
    plt.savefig(f'figures/{name}.pdf')
    plt.savefig(f'figures/{name}.png')
    plt.close()
    print(f'  Saved {name}')


def bar_panel(ax, labels, vals, errs, colors, ylabel, title, pval=None,
              rotation=0):
    x    = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5, width=0.55)
    ax.errorbar(x, vals, yerr=errs, fmt='none',
                color='#374151', linewidth=1.0, capsize=5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(errs) * 0.05,
                f'{val:.1f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#1F2937')  # was 9.5
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=rotation, ha='center')  # was 10
    ax.set_ylabel(ylabel, fontsize=13, labelpad=6)   # was 11
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10, loc='left')  # was 11
    ax.tick_params(axis='y', labelsize=12)           # was 10
    ax.grid(axis='y', linestyle='--', linewidth=0.4,
            color='#e5e7eb', zorder=0)
    ax.set_axisbelow(True)
    if pval is not None:
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01
                                           else ('*' if pval < 0.05 else 'n.s.'))
        ax.text(0.97, 0.97, f'$p$ = {pval:.3f}  {sig}',
                transform=ax.transAxes, fontsize=10,   # was 8.5
                ha='right', va='top', style='italic', color='#6B7280')


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Phase 1 vs Phase 2
# ════════════════════════════════════════════════════════════════════════════
def fig_phase_comparison(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))  # slightly taller
    specs = [
        ('tokens_sent',      1000, 'Mean tokens sent (thousands)'),
        ('api_calls',        1,    'Mean API calls (adjusted)'),
        ('duration_seconds', 1,    'Mean duration (seconds)'),
    ]
    titles = [
        'Token consumption\nPhase 1 vs Phase 2',
        'API call depth\nPhase 1 vs Phase 2',
        'Wall-clock duration\nPhase 1 vs Phase 2',
    ]
    for ax, (col, scale, ylabel), title in zip(axes, specs, titles):
        grp = (df.groupby('phase')[col]
                 .agg(['mean', 'std'])
                 .reindex(['P1', 'P2']))
        _, p = stats.ttest_ind(df[df['phase'] == 'P1'][col],
                               df[df['phase'] == 'P2'][col])
        bar_panel(ax,
                  labels=[PHASE_LABELS[p_] for p_ in grp.index],
                  vals=grp['mean'].values / scale,
                  errs=grp['std'].values / scale,
                  colors=[PHASE_COLORS[p_] for p_ in grp.index],
                  ylabel=ylabel, title=title, pval=p)
    plt.tight_layout(pad=2.0)
    savefig('fig_cost_phase_comparison')


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — Single-Agent vs Multi-Agent
# ════════════════════════════════════════════════════════════════════════════
def fig_single_vs_multi(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    specs = [
        ('tokens_sent',      1000, 'Mean tokens sent (thousands)'),
        ('api_calls',        1,    'Mean API calls (adjusted)'),
        ('duration_seconds', 1,    'Mean duration (seconds)'),
    ]
    titles = [
        'Token consumption\nSingle-Agent vs Multi-Agent',
        'API call depth\nSingle-Agent vs Multi-Agent',
        'Wall-clock duration\nSingle-Agent vs Multi-Agent',
    ]
    order = ['Single-Agent', 'Multi-Agent']
    for ax, (col, scale, ylabel), title in zip(axes, specs, titles):
        grp = (df.groupby('agent_class')[col]
                 .agg(['mean', 'std'])
                 .reindex(order))
        _, p = stats.ttest_ind(
            df[df['agent_class'] == 'Single-Agent'][col],
            df[df['agent_class'] == 'Multi-Agent'][col])
        bar_panel(ax, labels=order,
                  vals=grp['mean'].values / scale,
                  errs=grp['std'].values / scale,
                  colors=[CLASS_COLORS[c] for c in order],
                  ylabel=ylabel, title=title, pval=p)
    plt.tight_layout(pad=2.0)
    savefig('fig_cost_single_vs_multi')


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — Agent domain cost profile
# ════════════════════════════════════════════════════════════════════════════
def fig_agent_domain(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    order = ['IoT', 'FMSA', 'TSFM', 'WO', 'E2E']
    specs = [
        ('tokens_sent',      1000, 'Mean tokens sent (thousands)'),
        ('api_calls',        1,    'Mean API calls (adjusted)'),
        ('duration_seconds', 1,    'Mean duration (seconds)'),
    ]
    titles = [
        'Token consumption\nby agent domain',
        'API call depth\nby agent domain',
        'Wall-clock duration\nby agent domain',
    ]
    for ax, (col, scale, ylabel), title in zip(axes, specs, titles):
        grp = (df.groupby('agent')[col]
                 .agg(['mean', 'std'])
                 .reindex(order))
        bar_panel(ax, labels=order,
                  vals=grp['mean'].values / scale,
                  errs=grp['std'].values / scale,
                  colors=[AGENT_COLORS[a] for a in order],
                  ylabel=ylabel, title=title)
    plt.tight_layout(pad=2.0)
    savefig('fig_cost_agent_domain')


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — Phase × Agent heatmap
# ════════════════════════════════════════════════════════════════════════════
def fig_heatmap(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.0))
    agent_order = ['IoT', 'FMSA', 'TSFM', 'WO', 'E2E']
    phase_order = ['P1', 'P2']
    phase_tick  = ['Phase 1\n(Development)', 'Phase 2\n(Evaluation)']
    specs = [
        ('tokens_sent',      1000, 'Mean tokens (K)'),
        ('api_calls',        1,    'Mean API calls'),
        ('duration_seconds', 1,    'Mean duration (s)'),
    ]
    for ax, (col, scale, cbar_label) in zip(axes, specs):
        pivot = (df.groupby(['phase', 'agent'])[col]
                   .mean().unstack()
                   .reindex(index=phase_order, columns=agent_order) / scale)
        im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(agent_order)))
        ax.set_xticklabels(agent_order, fontsize=12)   # was 10
        ax.set_yticks(range(2))
        ax.set_yticklabels(phase_tick, fontsize=12)    # was 10
        vmax = np.nanmax(pivot.values)
        for i in range(2):
            for j in range(len(agent_order)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.1f}',
                            ha='center', va='center',
                            fontsize=11, fontweight='bold',   # was 9.5
                            color='white' if val > vmax * 0.6 else '#1F2937')
        cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
        cb.set_label(cbar_label, fontsize=11)          # was 9
        ax.set_title(cbar_label, fontsize=13,
                     fontweight='bold', pad=8, loc='left')
    plt.tight_layout(pad=2.0)
    savefig('fig_cost_heatmap')


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — Scenario difficulty split by phase
# ════════════════════════════════════════════════════════════════════════════
def fig_difficulty_cv(df):
    scen = (df.groupby(['id', 'agent', 'phase'])['tokens_sent']
              .agg(['mean', 'std', 'count'])
              .reset_index()
              .rename(columns={'mean':'mean_tokens',
                               'std': 'std_tokens',
                               'count':'n'}))
    scen['cv']    = scen['std_tokens'] / scen['mean_tokens']
    scen['label'] = scen['id'].map(lambda x: f'Q{x}')

    p1 = scen[scen['phase']=='P1'].sort_values('mean_tokens', ascending=True).reset_index(drop=True)
    p2 = scen[scen['phase']=='P2'].sort_values('mean_tokens', ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))  # taller for labels

    for ax, sub, title in zip(
        axes,
        [p1, p2],
        ['Token consumption per scenario\nPhase 1 — Development',
         'Token consumption per scenario\nPhase 2 — Evaluation'],
    ):
        colors = [AGENT_COLORS[a] for a in sub['agent']]
        ax.barh(sub['label'], sub['mean_tokens'] / 1000,
                color=colors, alpha=0.85, edgecolor='white', linewidth=0.4)
        ax.errorbar(sub['mean_tokens'] / 1000, sub['label'],
                    xerr=sub['std_tokens'] / 1000,
                    fmt='none', color='#6B7280', linewidth=0.8, capsize=2.5)
        for _, row in sub.iterrows():
            x_pos = (row['mean_tokens'] + row['std_tokens']) / 1000 + 3
            ax.text(x_pos, row['label'], row['agent'],
                    va='center', fontsize=10, color='#6B7280')  # was 8
        ax.set_xlabel('Mean tokens sent (thousands)', fontsize=13, labelpad=6)  # was 11
        ax.tick_params(labelsize=12)    # was 10
        ax.set_xlim(left=0)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10, loc='left')  # was 11
        ax.grid(axis='x', linestyle='--', linewidth=0.4,
                color='#e5e7eb', zorder=0)
        ax.set_axisbelow(True)
        ax.legend(handles=legend_agent, fontsize=11, frameon=True,   # was 9
                  framealpha=0.95, edgecolor='#e5e7eb', loc='lower right')

    plt.tight_layout(pad=2.5)
    savefig('fig_cost_difficulty_cv')


# ════════════════════════════════════════════════════════════════════════════
# FIG 6 — Cross-run boxplot both phases
# ════════════════════════════════════════════════════════════════════════════
def fig_boxplot_both_phases(df):
    PHASE1_IDS = [5, 8, 106, 114, 203, 204, 400, 405, 424, 604, 607]
    PHASE2_IDS = [7, 11, 107, 108, 201, 205, 403, 410, 411, 605, 606]

    combined_max = max(
        max(df[df['id']==sid]['tokens_sent'].max()/1000 for sid in PHASE1_IDS),
        max(df[df['id']==sid]['tokens_sent'].max()/1000 for sid in PHASE2_IDS),
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 6.0), sharey=True)

    for ax, phase_ids, phase_label, n_label in zip(
        axes,
        [PHASE1_IDS, PHASE2_IDS],
        ['Phase 1 (Development)', 'Phase 2 (Evaluation)'],
        [r'$n \approx 180$ executions per scenario',
         r'$n \approx 23$ executions per scenario'],
    ):
        order = sorted(phase_ids,
                       key=lambda x: df[df['id']==x]['tokens_sent'].mean(),
                       reverse=True)
        data   = [df[df['id']==sid]['tokens_sent'].values/1000 for sid in order]
        colors = [AGENT_COLORS[df[df['id']==sid]['agent'].iloc[0]] for sid in order]
        labels = [f"Q{sid}\n({df[df['id']==sid]['agent'].iloc[0]})" for sid in order]

        bp = ax.boxplot(data, vert=True, patch_artist=True,
                        medianprops=dict(color='white', linewidth=1.5),
                        whiskerprops=dict(linewidth=0.7, color='#6B7280'),
                        capprops=dict(linewidth=0.7, color='#6B7280'),
                        flierprops=dict(marker='o', markersize=2.5,
                                        markerfacecolor='#9CA3AF',
                                        alpha=0.5, linestyle='none'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.8); patch.set_linewidth(0.4)

        ax.set_xticks(range(1, len(order)+1))
        ax.set_xticklabels(labels, fontsize=11)        # was 9.5
        ax.set_ylabel('Tokens sent (thousands)', fontsize=13, labelpad=6)  # was 12
        ax.set_title(f'Cross-run token distribution — {phase_label}\n({n_label})',
                     fontsize=13, fontweight='bold', pad=10, loc='left')   # was 11
        ax.tick_params(axis='y', labelsize=12)         # was 10
        ax.grid(axis='y', linestyle='--', linewidth=0.4, color='#e5e7eb', zorder=0)
        ax.set_axisbelow(True)
        ax.legend(handles=legend_agent, fontsize=11, frameon=True,   # was 9.5
                  framealpha=0.95, edgecolor='#e5e7eb', loc='upper right')
        ax.set_ylim(0, combined_max * 1.08)

    plt.tight_layout(pad=2.0)
    savefig('fig_cost_boxplot_both_phases')


# ════════════════════════════════════════════════════════════════════════════
# FIG 7 — Execution cost distributions
# ════════════════════════════════════════════════════════════════════════════
def fig_experiment_distributions(df):
    p1 = df[df['phase'] == 'P1']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    specs = [
        ('tokens_sent',      1000, 'Tokens sent (thousands)',
         '#1D4ED8', 'Token consumption\nPhase 1 executions'),
        ('api_calls',        1,    'API calls (adjusted)',
         '#7C3AED', 'API call depth\nPhase 1 executions'),
        ('duration_seconds', 1,    'Duration (seconds)',
         '#D97706', 'Wall-clock duration\nPhase 1 executions'),
    ]
    for ax, (col, scale, xlabel, color, title) in zip(axes, specs):
        vals = p1[col] / scale
        ax.hist(vals, bins=35, color=color, alpha=0.8,
                edgecolor='white', linewidth=0.5)
        med = vals.median()
        ax.axvline(med, color='#DC2626', linewidth=1.2, linestyle='--')
        ylim = ax.get_ylim()
        ax.text(med * 1.03, ylim[1] * 0.88,
                f'Median\n{med:.1f}',
                fontsize=10, color='#DC2626', va='top')   # was 8.5
        ax.set_xlabel(xlabel, fontsize=13, labelpad=6)    # was 11
        ax.set_ylabel('Number of executions', fontsize=13, labelpad=6)  # was 11
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10, loc='left')  # was 11
        ax.tick_params(labelsize=12)   # was 10
        ax.grid(axis='y', linestyle='--', linewidth=0.4,
                color='#e5e7eb', zorder=0)
        ax.set_axisbelow(True)
    plt.tight_layout(pad=2.0)
    savefig('fig_cost_distributions')


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Loading data...')
    df = load_data('trajectory_stats.csv')
    print(f'  {len(df)} execution traces | '
          f'{df["id"].nunique()} scenarios | '
          f'{df["source_folder"].nunique()} unique experiment folders\n')

    print('Generating figures...')
    fig_phase_comparison(df)
    fig_single_vs_multi(df)
    fig_agent_domain(df)
    fig_heatmap(df)
    fig_difficulty_cv(df)
    fig_experiment_distributions(df)
    fig_boxplot_both_phases(df)

    print('\nAll figures saved to figures/')
    print('\n── Paper-ready numbers ──────────────────────────────────')
    for phase, label in [('P1','Phase 1'),('P2','Phase 2')]:
        sub = df[df['phase']==phase]
        print(f'{label}: {len(sub)} executions | '
              f'mean tokens={sub["tokens_sent"].mean():.0f} | '
              f'mean calls={sub["api_calls"].mean():.1f} | '
              f'mean dur={sub["duration_seconds"].mean():.1f}s')

    t, p = stats.ttest_ind(df[df['phase']=='P1']['tokens_sent'],
                           df[df['phase']=='P2']['tokens_sent'])
    print(f'\nPhase token t-test: t={t:.2f}, p={p:.4f}')

    sa = df[df['agent_class']=='Single-Agent']['tokens_sent']
    ma = df[df['agent_class']=='Multi-Agent']['tokens_sent']
    t, p = stats.ttest_ind(sa, ma)
    print(f'Single vs Multi-Agent tokens: '
          f'{sa.mean():.0f} vs {ma.mean():.0f} '
          f'({ma.mean()/sa.mean():.2f}x), t={t:.2f}, p={p:.4f}')

    top10 = (df[df['tokens_sent'] > df['tokens_sent'].quantile(0.9)]
             ['tokens_sent'].sum() / df['tokens_sent'].sum() * 100)
    print(f'Top 10% executions = {top10:.1f}% of total tokens')