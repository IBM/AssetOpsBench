"""
fig_benchmark_fingerprint.py
Input:  trajectory_stats.csv
Output: figures/fig_benchmark_fingerprint.pdf/.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["Georgia", "Times New Roman", "DejaVu Serif"],
    "font.size"        : 18,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.linewidth"   : 0.8,
    "figure.dpi"       : 300,
    "savefig.dpi"      : 300,
    "savefig.facecolor": "white",
    "savefig.bbox"     : "tight",
})

AGENT_COLORS = {
    'IoT' :'#2563EB','FMSR':'#7C3AED',
    'TSFM':'#059669','WO'  :'#D97706','E2E':'#DC2626',
}
PHASE1_IDS = [5,8,106,114,203,204,400,405,424,604,607]
PHASE2_IDS = [7,11,107,108,201,205,403,410,411,605,606]

def get_agent(sid):
    if sid in [5,7,8,11]: return 'IoT'
    if 100<=sid<=199:     return 'FMSR'
    if 200<=sid<=299:     return 'TSFM'
    if 400<=sid<=499:     return 'WO'
    if 600<=sid<=699:     return 'E2E'
    return 'Unknown'

df = pd.read_csv('trajectory_stats.csv')
df['api_calls'] = df['api_calls'] / 2
df['agent']     = df['id'].map(get_agent)
df['phase']     = df['id'].apply(
    lambda x: 'P1' if x in PHASE1_IDS else 'P2')

agents = ['IoT','FMSR','TSFM','WO','E2E']

domain_stats = {}
for agent in agents:
    sub   = df[df['agent']==agent]
    p1    = sub[sub['phase']=='P1']['tokens_sent']
    p2    = sub[sub['phase']=='P2']['tokens_sent']
    cv    = sub['tokens_sent'].std()/sub['tokens_sent'].mean()
    ratio = p2.mean()/p1.mean() if p1.mean()>0 else 1.0
    stab  = 1-abs(1-ratio)
    domain_stats[agent] = {
        'token_load'    : sub['tokens_sent'].mean()/1000,
        'predictability': stab/cv,
        'cv'            : cv,
        'stab'          : stab,
    }

fig, ax = plt.subplots(figsize=(10, 7.5))

ax.set_xlim(-10, 310)
ax.set_ylim(0.15, 1.55)

xs   = np.array([domain_stats[a]['token_load']     for a in agents])
ys   = np.array([domain_stats[a]['predictability'] for a in agents])
xmed = np.median(xs)
ymed = np.median(ys)

ax.axvline(xmed, color='#D1D5DB', linewidth=1.0,
           linestyle='--', zorder=1)
ax.axhline(ymed, color='#D1D5DB', linewidth=1.0,
           linestyle='--', zorder=1)

"""
for qx, qy, qtxt, ha, va in [
    (0.03, 0.97, 'Cheap &\nstrategy-sensitive',    'left',  'top'),
    (0.97, 0.57, 'Expensive &\nstrategy-sensitive', 'right', 'top'),
    (0.03, 0.03, 'Cheap &\npredictable',            'left',  'bottom'),
    (0.97, 0.03, 'Expensive &\npredictable',        'right', 'bottom'),
]:
    ax.text(qx, qy, qtxt, transform=ax.transAxes,
            fontsize=10, color='#9CA3AF',
            ha=ha, va=va, style='italic')
"""
                     
offsets = {
    'WO'  : ( 14,  10),
    'E2E' : ( 14,   8),
    'FMSA': ( 14,   8),
    'TSFM': (14, 8),
    'IoT' : ( 14, -22),
}

for agent in agents:
    x    = domain_stats[agent]['token_load']
    y    = domain_stats[agent]['predictability']
    c    = AGENT_COLORS[agent]
    ox,oy= offsets[agent]

    ax.scatter(x, y, s=300, color=c, alpha=0.92,
               edgecolors='white', linewidths=1.5, zorder=4)

    ax.annotate(
        f"$\\bf{{{agent}}}$\n"
        f"{x:.0f}K tok\n"
        f"CV={domain_stats[agent]['cv']:.2f}  "
        f"stab={domain_stats[agent]['stab']:.2f}",
        xy=(x, y), xytext=(ox, oy),
        textcoords='offset points',
        fontsize=9, color=c,
        arrowprops=dict(arrowstyle='-', color=c,
                        lw=0.8, alpha=0.6),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=c, linewidth=0.9, alpha=0.97),
        zorder=5,
    )

ax.text(0.50, 0.03,
        r'Predictability $= \frac{\mathrm{phase\ stability}}{\mathrm{CV}}$'
        '     '
        r'Phase stability $= 1 - |1 - \mu_{P2}/\mu_{P1}|$',
        transform=ax.transAxes, fontsize=8,
        ha='center', va='bottom', color='#6B7280',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#E5E7EB', linewidth=0.6, alpha=0.95))

ax.set_xlabel('Mean token load per execution (thousands)',
              fontsize=12, labelpad=8)
ax.set_ylabel(r'Predictability  (phase stability $\div$ CV)',
              fontsize=12, labelpad=8)
ax.tick_params(labelsize=11)
ax.grid(linestyle='--', linewidth=0.4, color='#F3F4F6', zorder=0)
ax.set_axisbelow(True)
ax.set_title(
    'Cost and predictability are inversely structured\nacross agent domains',
    fontsize=12, pad=12, loc='left',
    color='#111827', fontweight='bold',
)

plt.tight_layout()
plt.savefig('figures/fig_benchmark_fingerprint.pdf')
plt.savefig('figures/fig_benchmark_fingerprint.png')
plt.close()
print("Saved figures/fig_benchmark_fingerprint.pdf/.png")