import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family"        : "serif",
    "font.serif"         : ["Georgia", "Times New Roman", "DejaVu Serif"],
    "font.size"          : 14,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.linewidth"     : 0.7,
    "figure.dpi"         : 300,
    "savefig.dpi"        : 300,
    "savefig.facecolor"  : "white",
    "savefig.bbox"       : "tight",
})

track1_teams  = ["Team A", "Team B", "Team C", "Team D", "Team E", "Team F", "Team G", "Team H"]
track1_public = [72.73, 72.73, 71.43, 63.64, 63.64, 63.64, 54.55, 54.55]
track1_hidden = [54.55, 54.55, 45.45, 54.55, 54.55, 45.45, 54.55, 54.55]

"""
WaterLevel
BlueCube
Smart Maintenance Crew
LostSouls
Entropians
aviation_agent
Scalar_nitk
Infinity
kinatic
horizon
EXL Health AI Lab
"""

track2_teams  = ["Team A", "Team B", "Team C", "Team D", "Team E", "Team F", "Team G", "Team H"]
track2_public = [63.64, 45.45, 72.73, 45.45, 45.45, 54.55, 63.64, 54.55]
track2_hidden = [54.55, 63.64, 45.45, 63.64, 54.55, 63.64, 45.45, 54.55]

"""
Smart Maintenance Crew
WaterLevel
Scalar_nitk
horizon
LostSouls
Infinity
kinatic
BlueCube
Entropians
aviation_agent
EXL Health AI Lab
"""

C_PUBLIC  = "#2563EB"
C_HIDDEN  = "#059669"
C_SAT     = "#DC2626"
C_ANNOT   = "#374151"
X_MIN     = 30


def make_leaderboard(ax, teams, public, hidden, title, track_color,
                     show_saturation=False, sat_val=None):
    n     = len(teams)
    y     = np.arange(n)
    width = 0.36

    bars_p = ax.barh(y + width/2, public, width,
                     label="Public score",
                     color=track_color, alpha=0.85,
                     edgecolor="white", linewidth=0.5)
    bars_h = ax.barh(y - width/2, hidden, width,
                     label="Hidden score",
                     color=track_color, alpha=0.35,
                     edgecolor="white", linewidth=0.5,
                     hatch="///")

    # Public score labels
    for bar, val in zip(bars_p, public):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=10, color=track_color, fontweight="bold",
                clip_on=False)

    # Hidden score labels — clamped so short bars stay readable
    for bar, val in zip(bars_h, hidden):
        label_x = max(val + 0.5, X_MIN + 2.0)
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=10, color=C_ANNOT,
                clip_on=False)

    # Saturation line
    if show_saturation and sat_val:
        ax.axvline(sat_val, color=C_SAT, linewidth=1.2,
                   linestyle="--", zorder=5)
        ax.text(sat_val + 0.3, 1.01,
                f"Ceiling\n{sat_val:.1f}%",
                fontsize=9, color=C_SAT, va="bottom",
                transform=ax.get_xaxis_transform())

    ax.set_yticks(y)
    ax.set_yticklabels(teams, fontsize=11)
    ax.set_xlim(X_MIN, 85)
    ax.set_xlabel("Score (%)", fontsize=12, labelpad=6)
    ax.tick_params(axis="x", labelsize=11)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.set_title(title, fontsize=13, fontweight="bold",
                 pad=10, loc="left")
    #ax.grid(axis="x", linestyle="--",linewidth=0.5, color="#e5e7eb", zorder=0)
    ax.set_axisbelow(True)

    leg = ax.legend(frameon=True, fontsize=10,
                    loc="lower right", framealpha=0.95,
                    edgecolor="#e5e7eb", borderpad=0.7)
    leg.get_frame().set_linewidth(0.5)


# ════════════════════════════════════════════════════════════
# Figure 1 — Track 1 leaderboard
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.0, 4.2))
make_leaderboard(
    ax, track1_teams[::-1],
    track1_public[::-1], track1_hidden[::-1],
    title="",
    track_color=C_PUBLIC,
    show_saturation=True, sat_val=72.73
)
plt.tight_layout()
plt.savefig("figures/leaderboard_track1.pdf")
plt.savefig("figures/leaderboard_track1.png")
print("Saved leaderboard_track1")
plt.show()


# ════════════════════════════════════════════════════════════
# Figure 2 — Track 2 leaderboard
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.0, 4.2))
make_leaderboard(
    ax, track2_teams[::-1],
    track2_public[::-1], track2_hidden[::-1],
    title="",
    track_color="#7C3AED",
    show_saturation=False
)
plt.tight_layout()
plt.savefig("figures/leaderboard_track2.pdf")
plt.savefig("figures/leaderboard_track2.png")
print("Saved leaderboard_track2")
plt.show()


# ════════════════════════════════════════════════════════════
# Figure 3 — Participation funnel
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.0, 3.6))

stages = ["Registered", "Non-zero\npublic score",
          "Submitted\nboth tracks", "Fully\nranked"]
counts = [350, 24, 15, 11]
colors = ["#BFDBFE", "#93C5FD", "#60A5FA", "#2563EB"]

bars = ax.bar(stages, counts, color=colors,
              edgecolor="white", linewidth=0.8, width=0.55)

for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 4,
            str(val), ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#1E3A5F")

ax.set_ylabel("Number of teams", fontsize=12, labelpad=6)
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=11)
ax.set_ylim(0, 400)
ax.set_title("Participation Funnel", fontsize=13,
             fontweight="bold", pad=10, loc="left")
ax.text(0, 1.04,
        "350+ registrations · 11 fully ranked teams",
        transform=ax.transAxes, fontsize=9,
        color="#6B7280", style="italic")
#ax.grid(axis="y", linestyle="--",linewidth=0.5, color="#e5e7eb", zorder=0)
ax.set_axisbelow(True)
ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)

plt.tight_layout()
plt.savefig("figures/participation_funnel.pdf")
plt.savefig("figures/participation_funnel.png")
print("Saved participation_funnel")
plt.show()