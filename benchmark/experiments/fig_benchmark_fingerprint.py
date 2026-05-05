"""
fig_benchmark_fingerprint.py
Five radar charts — one per agent domain — clean, no table, no header.
Input:  trajectory_stats.csv
Output: figures/fig_fingerprint_main.pdf/.png
"""

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["Georgia", "Times New Roman", "DejaVu Serif"],
    "font.size"        : 15,         # was 13
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

agents      = ['WO','E2E','FMSR','IoT','TSFM']
axes_keys   = ['tokens','calls','duration','cv','stability']
axes_labels = [
    'Token\nload',
    'API call\ndepth',
    'Wall-clock\nduration',
    'Strategy\nvariance',
    'Phase\nstability',
]
fingerprints = {
    'WO'  : 'expensive · predictable · fair',
    'E2E' : 'cheap · stable · latency-dominant',
    'FMSR': 'moderate · reasoning-heavy · stable',
    'IoT' : 'moderate · strategy-sensitive · unstable',
    'TSFM': 'cheapest · most gameable · moderate',
}

raw = {}
for a in agents:
    sub  = df[df['agent']==a]
    p1   = sub[sub['phase']=='P1']['tokens_sent']
    p2   = sub[sub['phase']=='P2']['tokens_sent']
    cv   = sub['tokens_sent'].std() / sub['tokens_sent'].mean()
    stab = 1 - abs(1 - p2.mean()/p1.mean())
    raw[a] = {
        'tokens'   : sub['tokens_sent'].mean(),
        'calls'    : sub['api_calls'].mean(),
        'duration' : sub['duration_seconds'].mean(),
        'cv'       : cv,
        'stability': stab,
    }

norm = {}
for k in axes_keys:
    vals = np.array([raw[a][k] for a in agents])
    mn, mx = vals.min(), vals.max()
    norm[k] = {a: (raw[a][k]-mn)/(mx-mn) for a in agents}

def fmt(a, k):
    v = raw[a][k]
    return {
        'tokens'   : f"{v/1000:.0f}K tok",
        'calls'    : f"{v:.1f} calls",
        'duration' : f"{v:.0f}s",
        'cv'       : f"CV={v:.2f}",
        'stability': f"stab={v:.2f}",
    }[k]

N      = len(axes_keys)
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
ac     = np.append(angles, angles[0])

fig = plt.figure(figsize=(20, 8.5))   # wider + taller to accommodate larger fonts
gs  = GridSpec(1, 5, figure=fig,
               left=0.02, right=0.98,
               top=0.94,  bottom=0.04,
               wspace=0.10)            # slightly more space between panels

for col, agent in enumerate(agents):
    ax = fig.add_subplot(gs[col])
    c  = AGENT_COLORS[agent]
    v  = [norm[k][agent] for k in axes_keys]
    vc = v + [v[0]]

    # Grid rings
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(
            r * np.cos(np.linspace(0,2*np.pi,200)),
            r * np.sin(np.linspace(0,2*np.pi,200)),
            color='#CBD5E1' if r==1.0 else '#E2E8F0',
            linewidth=1.2 if r==1.0 else 0.5,
            zorder=1,
        )

    # Spokes
    for a in angles:
        ax.plot([0, np.cos(a)], [0, np.sin(a)],
                color='#E2E8F0', linewidth=0.5, zorder=1)

    # Filled polygon
    xs = [vv*np.cos(a) for vv,a in zip(vc, ac)]
    ys = [vv*np.sin(a) for vv,a in zip(vc, ac)]
    ax.fill(xs, ys, color=c, alpha=0.22, zorder=2)
    ax.plot(xs, ys, color=c, linewidth=2.5, zorder=3)
    ax.scatter(
        [vv*np.cos(a) for vv,a in zip(v, angles)],
        [vv*np.sin(a) for vv,a in zip(v, angles)],
        color=c, s=75, zorder=4,
        edgecolors='white', linewidths=1.2,
    )

    # Spoke labels with raw values
    for a, lbl, k in zip(angles, axes_labels, axes_keys):
        ax.text(
            1.44*np.cos(a), 1.44*np.sin(a),
            f"{lbl}\n{fmt(agent,k)}",
            ha='center', va='center',
            fontsize=16,              # was 10.5
            color='#334155',
            bbox=dict(boxstyle='round,pad=0.22',
                      facecolor='white',
                      edgecolor='none', alpha=0.93),
        )

    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.85, 1.85)
    ax.set_aspect('equal')
    ax.axis('off')

    # Domain name
    ax.text(0, 1.85, agent,
            ha='center', va='bottom',
            fontsize=19, fontweight='bold', color=c)   # was 16

    """
    # Fingerprint tag directly under domain name
    ax.text(0, 1.67, f"({fingerprints[agent]})",
            ha='center', va='bottom',
            fontsize=14, color=c, style='italic')    # was 10
    """
    
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_fingerprint_main.pdf')
plt.savefig('figures/fig_fingerprint_main.png')
plt.close()
print("Saved figures/fig_fingerprint_main.pdf/.png")