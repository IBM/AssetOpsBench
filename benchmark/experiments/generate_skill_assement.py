"""
fig_archetype_bars.py
Output: figures/fig_archetype_bars.pdf/.png
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["Georgia", "Times New Roman", "DejaVu Serif"],
    "font.size"        : 13,
    "figure.dpi"       : 300,
    "savefig.dpi"      : 300,
    "savefig.facecolor": "white",
    "savefig.bbox"     : "tight",
})

T1_COLOR = '#2563EB'
T2_COLOR = '#D97706'

track1 = [
    ('P1','KB-grounded plan',        32.4,0.854,2,'knowledge base · agent'),
    ('P2','Reviewer checklist',      25.2,0.853,1,'reviewer · checklist · enforcement'),
    ('P3','Tool-semantics two-pass', 22.4,0.850,3,'tools · second prompt · semantics'),
    ('P4','Step-sequential parser',  11.4,0.793,0,'step · sequential · dynamic examples'),
    ('P5','Agent-catalog rules',      8.6,0.804,0,'agent catalog · USE-THIS-WHEN'),
]
track2 = [
    ('E1','Soft-validation fallback', 28.9,0.850,3,'valueerror · proceed · fallback'),
    ('E2','External-LLM refinement',  24.8,0.881,2,'external llm · watsonx · max_retries'),
    ('E3','Keyword-overlap executor', 19.0,0.837,2,'keyword overlap · stopword · score'),
    ('E4','Forbidden-word filter',    15.7,0.851,1,'dataset · forbidden word · fallback'),
    ('E5','Grounded consensus',       11.6,0.842,2,'verify_grounding · circuit · consensus'),
]

def draw_track(ax, data, color, title, subtitle):
    n = len(data)
    y = np.arange(n)[::-1]

    for i,(code,name,share,coh,stab,terms) in enumerate(data):
        yi = y[i]

        # Solid bar — no opacity
        ax.barh(yi, share, height=0.62,
                color=color, alpha=1.0,
                edgecolor='white', linewidth=0.8)

        # Share % inside bar
        ax.text(share-0.8, yi, f"{share:.1f}%",
                va='center', ha='right',
                fontsize=12, fontweight='bold', color='white')

        # Code badge
        ax.text(-0.8, yi, code,
                va='center', ha='right',
                fontsize=14, fontweight='bold', color=color)

        # Archetype name
        ax.text(share+0.7, yi+0.19, name,
                va='bottom', ha='left',
                fontsize=14, color='#0F172A', fontweight='bold')

        # Characteristic terms
        ax.text(share+0.7, yi-0.19, terms,
                va='top', ha='left',
                fontsize=12, color='#475569', style='italic')

        # Stability dots
        for d in range(5):
            ax.plot(share+0.5+d*0.9, yi, 'o',
                    color=color if d<stab else '#CBD5E1',
                    markersize=6.5, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.5)

        # Cohesion
        ax.text(36.5, yi, f"{coh:.3f}",
                va='center', ha='left',
                fontsize=12, color='#64748B', fontweight='bold')

    # Column headers
    for lbl, xpos, ha in [
        ('ID',    -0.8, 'right'),
        ('Share',  0.8, 'left'),
        ('Stability',  4.8, 'left'),
        ('Cohesion',  36.5, 'left'),
    ]:
        ax.text(xpos, n+0.15, lbl, va='bottom', ha=ha,
                fontsize=10, color='#64748B', fontweight='bold')

    ax.axhline(n-0.38, color='#CBD5E1', linewidth=0.8)
    ax.set_xlim(-3, 41)
    ax.set_ylim(-0.65, n+0.45)
    ax.set_yticks([]); ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{title}\n{subtitle}",
                 fontsize=13, fontweight='bold',
                 color=color, pad=10, loc='left')

fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
fig.subplots_adjust(wspace=0.18, left=0.01, right=0.99,
                    top=0.87, bottom=0.10)

draw_track(axes[0], track1, T1_COLOR,
           'Track 1 — Planning',
           '5 archetypes · top-3 cover 80.0%')
draw_track(axes[1], track2, T2_COLOR,
           'Track 2 — Execution',
           '5 archetypes · top-4 cover 88.4%')

fig.add_artist(plt.Line2D(
    [0.505, 0.505], [0.08, 0.93],
    transform=fig.transFigure,
    color='#E2E8F0', linewidth=1.0))

fig.legend(
    handles=[
        mpatches.Circle((0,0), radius=0.05, color='#475569',
                         label='Encoder-stable medoid'),
        mpatches.Circle((0,0), radius=0.05, color='#CBD5E1',
                         label='Not stable across encoders'),
    ],
    loc='lower center', ncol=2, fontsize=14,
    frameon=True, framealpha=0.95, edgecolor='#E2E8F0',
    bbox_to_anchor=(0.5, 0.0))

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig_archetype_bars.pdf')
plt.savefig('figures/fig_archetype_bars.png')
plt.close()
print("Saved figures/fig_archetype_bars.pdf/.png")