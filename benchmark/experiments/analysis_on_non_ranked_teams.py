"""
analysis_on_non_ranked_teams.py
Analysis on non_ranked teams for the NeurIPS evaluation paper.
Input:  Best Submissions of codabench competition
Output: figures
"""

import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("Best_Submissions.xlsx")
df["TeamName"] = df["TeamName"].str.strip().str.lower()
# =========================
# BASIC CLEAN
# =========================
df.columns = df.columns.str.strip()

# =========================
# FILTER ONLY FINISHED
# =========================
df = df[df['Status'] == 'Finished']

# =========================
# REMOVE EMPTY SCORES
# =========================
df = df.dropna(subset=['Score'])

# =========================
# GROUP BY TEAM + TRACK
# =========================
# get best score per team per track
team_track = df.groupby(['TeamName', 'Track'])['Score'].max().unstack()

# rename columns for clarity
team_track.columns = ['Execution', 'Planning'] if 'Task Execution' in team_track.columns else team_track.columns

print("\n=== TEAM TRACK TABLE ===")
print(team_track)

# =========================
# CREATE FINAL TABLE
# =========================
team_track = team_track.fillna(0)

# =========================
# METRIC 1: TOTAL SCORE (proxy)
# =========================
team_track['Total'] = 0.4 * team_track.get('Planning', 0) + 0.6 * team_track.get('Execution', 0)

# =========================
# METRIC 2: IMBALANCE
# =========================
team_track['imbalance'] = abs(team_track.get('Planning', 0) - team_track.get('Execution', 0))

# =========================
# RANK TEAMS
# =========================
team_track = team_track.sort_values('Total', ascending=False)
team_track['rank'] = range(1, len(team_track)+1)

print("\n=== FINAL TEAM TABLE ===")
print(team_track)

# =========================
# SPLIT TOP vs OTHERS
# =========================
top = team_track.head(11)
others = team_track.iloc[11:]

# =========================
# SUMMARY STATS
# =========================
def summarize(group, name):
    print(f"\n=== {name} ===")
    print("Mean Planning:", group['Planning'].mean())
    print("Mean Execution:", group['Execution'].mean())
    print("Mean Total:", group['Total'].mean())
    print("Mean Imbalance:", group['imbalance'].mean())

summarize(top, "TOP TEAMS")
summarize(others, "NEAR-MISS TEAMS")

# =========================
# CORRELATION
# =========================
corr = team_track[['Planning','Execution']].corr().iloc[0,1]
print("\nPlanning vs Execution correlation:", corr)

# =========================
# THRESHOLD
# =========================
threshold = top['Total'].min()
print("\nRanking threshold:", threshold)

# =========================
# OPTIONAL: PRINT TEAM LISTS
# =========================
print("\nTop Teams:\n", top.index.tolist())
print("\nNear-Miss Teams:\n", others.index.tolist())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("Best_Submissions.xlsx")
df.columns = df.columns.str.strip()
df["TeamName"] = df["TeamName"].str.strip().str.lower()

# =========================
# FILTER
# =========================
df = df[df["Status"] == "Finished"].dropna(subset=["Score"])

# Optional: inspect actual track labels once
print("Unique tracks:", df["Track"].dropna().unique())

# =========================
# GROUP BY TEAM + TRACK
# =========================
team_track = df.groupby(["TeamName", "Track"])["Score"].max().unstack()

# Rename track names explicitly if needed
team_track = team_track.rename(columns={
    "Task Execution": "Execution",
    "Task Planning": "Planning",
    # add any other raw labels here if needed
})

# Keep only the two tracks you care about, if both exist
for col in ["Planning", "Execution"]:
    if col not in team_track.columns:
        team_track[col] = np.nan

team_track = team_track[["Planning", "Execution"]].fillna(0)

# =========================
# METRICS
# =========================
team_track["Total"] = 0.4 * team_track["Planning"] + 0.6 * team_track["Execution"]
team_track["imbalance"] = (team_track["Planning"] - team_track["Execution"]).abs()
team_track = team_track.sort_values("Total", ascending=False)
team_track["rank"] = range(1, len(team_track) + 1)

top = team_track.head(11)
others = team_track.iloc[11:]
threshold = top["Total"].min()

print(team_track.head(15))

# =========================
# STYLE
# =========================
sns.set_theme(style="whitegrid")

# =========================
# PLOT 1: TOTAL SCORE BY TEAM
# =========================
plt.figure(figsize=(12, 8))
plot_df = team_track.reset_index()

sns.barplot(
    data=plot_df,
    y="TeamName",
    x="Total",
    color="steelblue"
)

plt.axvline(threshold, linestyle="--", color="red", label=f"Top-11 threshold = {threshold:.2f}")
plt.title("Team Ranking by Weighted Total Score")
plt.xlabel("Weighted Total Score")
plt.ylabel("Team")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# PLOT 2: PLANNING vs EXECUTION SCATTER
# Color = imbalance, size = total
# =========================
plt.figure(figsize=(8, 7))
scatter_df = team_track.reset_index()

sizes = 80 + 400 * (scatter_df["Total"] - scatter_df["Total"].min()) / (
    scatter_df["Total"].max() - scatter_df["Total"].min() + 1e-9
)

sc = plt.scatter(
    scatter_df["Planning"],
    scatter_df["Execution"],
    s=sizes,
    c=scatter_df["imbalance"],
    cmap="viridis",
    alpha=0.8,
    edgecolor="black"
)

plt.plot(
    [scatter_df["Planning"].min(), scatter_df["Planning"].max()],
    [scatter_df["Planning"].min(), scatter_df["Planning"].max()],
    linestyle="--",
    color="gray",
    label="Balanced line"
)

plt.colorbar(sc, label="Imbalance |Planning - Execution|")
plt.title("Planning vs Execution Tradeoff")
plt.xlabel("Planning Score")
plt.ylabel("Execution Score")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# PLOT 3: TOP TEAMS VS OTHERS
# Boxplot for Total, Planning, Execution, Imbalance
# =========================
compare_df = pd.concat([
    top.assign(Group="Top 11"),
    others.assign(Group="Others")
]).reset_index()

metrics = ["Planning", "Execution", "Total", "imbalance"]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.ravel()

for ax, metric in zip(axes, metrics):
    sns.boxplot(data=compare_df, x="Group", y=metric, ax=ax)
    ax.set_title(f"{metric}: Top 11 vs Others")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()

# =========================
# PLOT 4: HEATMAP OF PLANNING/EXECUTION
# Show top teams only for readability
# =========================
plt.figure(figsize=(8, 10))
heat_df = team_track[["Planning", "Execution"]].head(20)

sns.heatmap(
    heat_df,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title("Planning and Execution Scores for Top 20 Teams")
plt.xlabel("Track")
plt.ylabel("Team")
plt.tight_layout()
plt.show()

# =========================
# OPTIONAL: SAVE FIGURES
# =========================
# plt.savefig("plot_name.png", dpi=300, bbox_inches="tight")