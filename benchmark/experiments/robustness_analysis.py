"""
robustness_analysis.py
Analysis on robustness of submissions for the NeurIPS evaluation paper.
Input:  Final CODS results of codabench competition
Output: figures
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()   # or Path.cwd() / 'comp' if needed
FINAL_XLSX = ROOT / 'Final CODS Results.xlsx'
REG_CSV = ROOT / 'AssetOpsBench AI Challenge.csv'
OUTDIR = ROOT / 'assetopsbench_appendix_figures'
OUTDIR.mkdir(exist_ok=True)

print("Working directory:", ROOT)
print("XLSX:", FINAL_XLSX)
print("CSV:", REG_CSV)


plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 160,
    'savefig.dpi': 220,
})


def load_results() -> pd.DataFrame:
    df = pd.read_excel(FINAL_XLSX)
    df = df.dropna(subset=['TeamName']).copy()
    num_cols = [
        'Score_Public_Execution', 'Score_Public_Planning',
        'Score_Private_Execution', 'Score_Private_Planning',
        't_match_execution', 't-match_planning',
        'Composite_Track1', 'Composite_Track2', 'Final Rank'
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_registrations() -> pd.DataFrame:
    df = pd.read_csv(REG_CSV)
    df['_ts'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    team_col = 'Team Name (same as you would create on the submission platform)'
    username_col = 'Codaench username of each team member separated by comma'
    df = df.sort_values('_ts')
    latest = df.groupby(df[team_col].astype(str).str.strip(), as_index=False).tail(1).copy()
    def split_list(x):
        if pd.isna(x) or not str(x).strip():
            return []
        return [p.strip() for p in str(x).split(',') if p.strip()]
    latest['username_count'] = latest[username_col].apply(lambda x: len(split_list(x)))
    latest['num_members'] = pd.to_numeric(latest['Number of Team Members :'], errors='coerce')
    return latest


def compute_final(df: pd.DataFrame, tscale: float = 1.0, w_exec: float = 0.6) -> pd.DataFrame:
    w_plan = 1.0 - w_exec
    out = df[['TeamName']].copy()
    out['C_plan'] = 0.6 * df['Score_Public_Planning'] + 0.3 * df['Score_Private_Planning'] + 0.1 * tscale * df['t-match_planning']
    out['C_exec'] = 0.6 * df['Score_Public_Execution'] + 0.3 * df['Score_Private_Execution'] + 0.1 * tscale * df['t_match_execution']
    out['F'] = w_plan * out['C_plan'] + w_exec * out['C_exec']
    out = out.sort_values('F', ascending=False).reset_index(drop=True)
    out['rank'] = np.arange(1, len(out) + 1)
    return out


def add_team_labels(ax, x, y, labels, dx=0.0, dy=0.0, max_labels=4):
    # label the points farthest from y=x or farthest from the median if used on a scatter plot
    if len(x) == 0:
        return
    idx = np.argsort(np.abs(y - x))[::-1][:max_labels]
    for i in idx:
        ax.annotate(labels[i], (x[i], y[i]), xytext=(5, 5), textcoords='offset points', fontsize=9,
                    arrowprops=dict(arrowstyle='-', lw=0.6, alpha=0.5))


def fig_public_hidden(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
    panels = [
        ('Score_Public_Planning', 'Score_Private_Planning', 'Planning'),
        ('Score_Public_Execution', 'Score_Private_Execution', 'Execution')
    ]
    for ax, (xcol, ycol, title) in zip(axes, panels):
        x = df[xcol].to_numpy()
        y = df[ycol].to_numpy()
        # color by absolute rank shift (public -> private) within track
        pub_rank = pd.Series(-x).rank(method='min').to_numpy()
        priv_rank = pd.Series(-y).rank(method='min').to_numpy()
        shift = np.abs(pub_rank - priv_rank)
        sc = ax.scatter(x, y, c=shift, s=90, cmap='viridis', edgecolor='white', linewidth=0.8)
        lo = min(x.min(), y.min()) - 1
        hi = max(x.max(), y.max()) + 1
        ax.plot([lo, hi], [lo, hi], '--', color='0.35', lw=1)
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.03, 0.96, f'r = {r:.2f}', transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.85'))
        add_team_labels(ax, x, y, df['TeamName'].tolist(), max_labels=4)
        ax.set_title(f'{title}: public vs hidden score')
        ax.set_xlabel('Public score')
        ax.set_ylabel('Hidden score')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.2)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label('Approx. rank shift magnitude')
    fig.suptitle('Hidden evaluation changes the story', y=1.03, fontsize=15)
    fig.savefig(OUTDIR / 'fig1_public_hidden_scatter.png', bbox_inches='tight')
    plt.close(fig)


def fig_winner_stability(df: pd.DataFrame):
    # grid over t-match scaling and execution weight
    scales = np.array([1, 3, 10, 30, 100], dtype=float)
    w_execs = np.linspace(0.1, 0.9, 17)
    winners = np.empty((len(w_execs), len(scales)), dtype=int)
    margins = np.empty_like(winners, dtype=float)
    base = compute_final(df, 1, 0.6)
    team_order = base['TeamName'].tolist()  # canonical order by published ranking
    team_to_idx = {t: i for i, t in enumerate(team_order)}

    for i, w in enumerate(w_execs):
        for j, s in enumerate(scales):
            res = compute_final(df, s, w)
            winners[i, j] = team_to_idx[res.iloc[0]['TeamName']]
            margins[i, j] = res.iloc[0]['F'] - res.iloc[1]['F']

    cmap = ListedColormap(plt.get_cmap('tab10').colors + plt.get_cmap('tab20').colors)
    norm = BoundaryNorm(np.arange(-0.5, len(team_order) + 0.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(winners, origin='lower', aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f'{s:g}x' for s in scales])
    ax.set_yticks(range(len(w_execs))[::2])
    ax.set_yticklabels([f'{w:.1f}' for w in w_execs[::2]])
    ax.set_xlabel('t-match rescaling factor')
    ax.set_ylabel('Execution weight')
    ax.set_title('Winner identity under score perturbations')
    # annotate cells with initials
    initials = {t: ''.join([w[0] for w in t.split()][:2]).upper() for t in team_order}
    for i in range(len(w_execs)):
        for j in range(len(scales)):
            t = team_order[winners[i, j]]
            ax.text(j, i, initials[t], ha='center', va='center', fontsize=8,
                    color='white' if winners[i, j] in {0, 1, 3, 5, 7, 8} else 'black')

    # legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=cmap(i), label=team_order[i]) for i in range(len(team_order))]
    ax.legend(handles=patches[:6], loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, title='Top winners')

    ax2 = axes[1]
    im2 = ax2.imshow(margins, origin='lower', aspect='auto', cmap='magma')
    ax2.set_xticks(range(len(scales)))
    ax2.set_xticklabels([f'{s:g}x' for s in scales])
    ax2.set_yticks(range(len(w_execs))[::2])
    ax2.set_yticklabels([f'{w:.1f}' for w in w_execs[::2]])
    ax2.set_xlabel('t-match rescaling factor')
    ax2.set_ylabel('Execution weight')
    ax2.set_title('Top-1 margin in the same grid')
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label('Score margin (Top-1 minus Top-2)')
    fig.suptitle('Published ranking is stable only in a narrow parameter region', y=1.03, fontsize=15)
    fig.savefig(OUTDIR / 'fig2_winner_stability_heatmap.png', bbox_inches='tight')
    plt.close(fig)


def fig_saturation(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    score_cols = [('Score_Public_Planning', 'Planning'), ('Score_Public_Execution', 'Execution')]
    bins = np.arange(25, 80, 1.8)
    for ax, (col, title) in zip(axes, score_cols):
        s = df[col].dropna().to_numpy()
        vals, counts = np.unique(np.round(s, 2), return_counts=True)
        ax.bar(vals, counts, width=0.8, alpha=0.9)
        ax.set_title(f'Public score saturation: {title}')
        ax.set_xlabel('Public score')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.2)
        ax.axvline(vals.max(), ls='--', lw=1, color='0.35')
        ax.text(0.98, 0.95, f'{len(vals)} distinct values\nceiling = {vals.max():.2f}', transform=ax.transAxes,
                ha='right', va='top', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.85'))
    fig.suptitle('The public leaderboard is coarse and saturates early', y=1.03, fontsize=15)
    fig.savefig(OUTDIR / 'fig3_public_score_saturation.png', bbox_inches='tight')
    plt.close(fig)


def fig_registration_friction(reg: pd.DataFrame):
    team_col = 'Team Name (same as you would create on the submission platform)'
    roles = ['Student [Undergraduate]', 'Industry Professional', 'Student [Masters or Phd]', 'Other']
    prof_cols = [c for c in reg.columns if c.startswith('Profession [Member')]

    # latest registration per team already selected
    # team size / usernames
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)

    ax = axes[0]
    counts = reg['username_count'].fillna(0).astype(int)
    vals, cts = np.unique(counts, return_counts=True)
    ax.bar(vals, cts, width=0.75)
    ax.set_xlabel('# Codabench usernames listed in team form')
    ax.set_ylabel('# teams')
    ax.set_title('Account fragmentation in registration')
    ax.grid(axis='y', alpha=0.2)
    ax.text(0.98, 0.95, f'mean = {counts.mean():.2f}\n>1 usernames = {(counts>1).sum()}/{len(counts)}',
            transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.85'))

    ax = axes[1]
    # role counts from all member fields
    role_counts = {}
    for c in prof_cols:
        for v in reg[c].dropna().astype(str):
            v = v.strip()
            if not v:
                continue
            role_counts[v] = role_counts.get(v, 0) + 1
    # ordered bars
    ordered = [(r, role_counts.get(r, 0)) for r in roles]
    ax.bar([o[0] for o in ordered], [o[1] for o in ordered])
    ax.set_title('Participant composition')
    ax.set_ylabel('Declared member slots')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.2)
    fig.suptitle('Registration friction and participant heterogeneity', y=1.03, fontsize=15)
    fig.savefig(OUTDIR / 'fig4_registration_friction.png', bbox_inches='tight')
    plt.close(fig)


def fig_rank_shift_hist(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for ax, (xcol, ycol, title) in zip(axes, [
        ('Score_Public_Planning', 'Score_Private_Planning', 'Planning'),
        ('Score_Public_Execution', 'Score_Private_Execution', 'Execution'),
    ]):
        pub_rank = pd.Series(-df[xcol]).rank(method='min').to_numpy()
        priv_rank = pd.Series(-df[ycol]).rank(method='min').to_numpy()
        shift = priv_rank - pub_rank  # positive = hidden rank worse than public
        ax.hist(shift, bins=np.arange(shift.min()-0.5, shift.max()+1.5, 1), edgecolor='black')
        ax.axvline(0, color='0.35', ls='--', lw=1)
        ax.set_title(f'{title}: rank shift (hidden - public)')
        ax.set_xlabel('Rank shift')
        ax.set_ylabel('Teams')
        ax.grid(axis='y', alpha=0.2)
        ax.text(0.98, 0.95, f'median = {np.median(shift):.0f}\nmax |shift| = {np.max(np.abs(shift)):.0f}',
                transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.85'))
    fig.suptitle('Hidden evaluation reorders teams', y=1.03, fontsize=15)
    fig.savefig(OUTDIR / 'fig5_rank_shift_hist.png', bbox_inches='tight')
    plt.close(fig)


def main():
    df = load_results()
    reg = load_registrations()
    fig_public_hidden(df)
    fig_winner_stability(df)
    fig_saturation(df)
    fig_registration_friction(reg)
    fig_rank_shift_hist(df)
    print('Saved figures to', OUTDIR)

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("/content/assetops_clean_analysis.csv")

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)

# =========================
# 2. BASIC CLEANING
# =========================
# Convert booleans if needed
bool_cols = ['has_loop', 'has_errors', 'has_repetition']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

# =========================
# 3. EXPERIMENT 1: Behavior vs Success
# =========================
print("\n=== Behavior vs Task Success ===")

features = [
    'num_steps', 'num_actions', 'num_tool_actions',
    'tool_entropy', 'tokens_sent', 'api_calls',
    'error_rate', 'repeat_rate'
]

for f in features:
    if f in df.columns:
        print(f"\n{f}")
        print(df.groupby('task_success')[f].mean())

# =========================
# 4. EXPERIMENT 2: Failure Type Bias
# =========================
if 'failure_type' in df.columns:
    print("\n=== Failure Type Bias ===")
    print(df.groupby('failure_type')['task_success'].mean())

# =========================
# 5. EXPERIMENT 3: Loop / Error Bias
# =========================
print("\n=== Loop / Error Bias ===")

for col in ['has_loop', 'has_errors', 'has_repetition']:
    if col in df.columns:
        print(f"\n{col}")
        print(df.groupby(col)['task_success'].mean())

# =========================
# 6. EXPERIMENT 4: Correlation Analysis
# =========================
print("\n=== Correlation with Task Success ===")

corr = df.corr(numeric_only=True)
print(corr['task_success'].sort_values(ascending=False))

# =========================
# 7. PLOTS
# =========================
sns.set(style="whitegrid")

# --- Plot 1: Tokens vs Success
if 'tokens_sent' in df.columns:
    plt.figure()
    sns.boxplot(x='task_success', y='tokens_sent', data=df)
    plt.title("Token Usage vs Task Success")
    plt.savefig("tokens_vs_success.png")
    plt.show()

# --- Plot 2: Steps vs Success
if 'num_steps' in df.columns:
    plt.figure()
    sns.boxplot(x='task_success', y='num_steps', data=df)
    plt.title("Number of Steps vs Task Success")
    plt.savefig("steps_vs_success.png")
    plt.show()

# --- Plot 3: Tool entropy vs Success
if 'tool_entropy' in df.columns:
    plt.figure()
    sns.boxplot(x='task_success', y='tool_entropy', data=df)
    plt.title("Tool Entropy vs Task Success")
    plt.savefig("entropy_vs_success.png")
    plt.show()

# --- Plot 4: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# =========================
# 8. SIMPLE INTERPRETATION PRINT
# =========================
print("\n=== QUICK INTERPRETATION ===")

print("""
Check:
- If tokens_sent is much higher for success → possible verbosity bias
- If num_steps strongly correlates → structure bias
- If error_rate strongly negative → correct penalty
- If correlations are weak → evaluation is robust

Key goal:
→ Show that success depends on correctness, NOT superficial features
""")