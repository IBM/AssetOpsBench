"""
AssetOpsBench — Figure Generation Script (v2)
==============================================
Changes from v1:
  - White / transparent backgrounds (no coloured panels)
  - Larger fonts throughout for readability
  - Radar chart removed; heatmap promoted to standalone Figure 3
  - Team names replaced with encoded IDs (Team A … Team K)
    matching the anonymisation table in the paper

Requirements:
    pip install pandas matplotlib openpyxl

Input:
    submissions_with_team_final_with_track.csv

Output (written to ./figures/):
    fig1_overview.{png,pdf}
    fig2_trajectory.{png,pdf}
    fig3_heatmap.{png,pdf}
    fig4_learning_success.{png,pdf}

Usage:
    python generate_figures.py
    python generate_figures.py --data path/to/file.csv --out path/to/figures/
"""

import argparse, os, warnings
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np, pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="submissions_with_team_final_with_track.csv")
parser.add_argument("--out",  default="figures")
parser.add_argument("--dpi",  type=int, default=180)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(args.data)
df["TeamName"]      = df["TeamName"].str.strip()
df["TeamName_norm"] = df["TeamName"].str.lower()
df["Date"]          = pd.to_datetime(df["Date"], format="%Y-%m-%d_%H:%M")
fin = df[df["Status"] == "Finished"].dropna(subset=["Score"]).copy()

# ── TEAM MAPPING ──────────────────────────────────────────────────────────────
TEAM_META = {
    "waterlevel":             ("Team A", "#FFD700"),
    "bluecube":               ("Team B", "#C0C0C0"),
    "smart maintenance crew": ("Team C", "#CD7F32"),
    "entropians":             ("Team D", "#4C72B0"),
    "aviation_agent":         ("Team E", "#DD8452"),
    "lostsouls":              ("Team F", "#55A868"),
    "scalar_nitk":            ("Team G", "#C44E52"),
    "infinity":               ("Team H", "#8172B2"),
    "kinatic":                ("Team I", "#937860"),
    "horizon":                ("Team J", "#DA8BC3"),
    "exl health ai lab":      ("Team K", "#8C8C8C"),
}
RANKED    = list(TEAM_META.keys())
PLAN_COL  = "#3A7DCC"
EXEC_COL  = "#E05C2A"
GREY      = "#222222"
FS_SUPER, FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT = 16, 14, 13, 11, 10

plt.rcParams.update({
    "font.size": FS_TICK, "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK, "legend.fontsize": FS_ANNOT,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

all_teams = [t for t in RANKED if t in fin["TeamName_norm"].unique()]
labels    = [TEAM_META[t][0] for t in all_teams]
fin["color"] = fin["TeamName_norm"].map(lambda x: TEAM_META.get(x,("?","#BBB"))[1])
fin["label"] = fin["TeamName_norm"].map(lambda x: TEAM_META.get(x,("?","?"))[0])
best = fin.groupby(["TeamName_norm","Track"])["Score"].max().unstack(); best.columns.name=None
cnts = fin.groupby(["TeamName_norm","Track"])["Score"].count().unstack(fill_value=0); cnts.columns.name=None

def style_ax(ax, xgrid=False):
    ax.set_facecolor("white"); ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.40, color="#CCCCCC", zorder=0)
    if xgrid: ax.xaxis.grid(True, linestyle="--", alpha=0.40, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)

def save_fig(fig, name):
    base = os.path.join(args.out, name)
    fig.savefig(base+".png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base+".pdf",              bbox_inches="tight", facecolor="white")
    print(f"  Saved {base}.png / .pdf"); plt.close(fig)

# ── FIGURE 1 ──────────────────────────────────────────────────────────────────
def figure1():
    fig = plt.figure(figsize=(22,14), facecolor="white")
    gs  = gridspec.GridSpec(2,2, figure=fig, hspace=0.42, wspace=0.28,
                             left=0.06, right=0.97, top=0.90, bottom=0.09)
    ax_box = fig.add_subplot(gs[0,:]); ax_sc = fig.add_subplot(gs[1,0]); ax_bar = fig.add_subplot(gs[1,1])

    w, gap = 0.35, 0.20
    pos_plan, pos_exec = [], []
    for i in range(len(all_teams)):
        cx = i*(2*w+gap+0.20); pos_plan.append(cx-w/2); pos_exec.append(cx+w/2)

    for positions, track, tc in zip([pos_plan,pos_exec],["Task Planning","Task Execution"],[PLAN_COL,EXEC_COL]):
        data = [fin[(fin["TeamName_norm"]==t)&(fin["Track"]==track)]["Score"].dropna().values for t in all_teams]
        data = [d if len(d)>0 else np.array([np.nan]) for d in data]
        bp = ax_box.boxplot(data, positions=positions, widths=w, patch_artist=True,
                            medianprops=dict(color="white",linewidth=2.2),
                            whiskerprops=dict(linewidth=1.4,color="#555"),
                            capprops=dict(linewidth=1.4,color="#555"),
                            flierprops=dict(marker="o",markersize=4,alpha=0.55,markerfacecolor=tc,markeredgewidth=0),
                            manage_ticks=False)
        for p in bp["boxes"]: p.set(facecolor=tc,alpha=0.82,edgecolor="#333",linewidth=0.9)

    for i,t in enumerate(all_teams):
        for pos,track in zip([pos_plan[i],pos_exec[i]],["Task Planning","Task Execution"]):
            v = fin[(fin["TeamName_norm"]==t)&(fin["Track"]==track)]["Score"].dropna()
            if len(v): ax_box.plot(pos,v.mean(),"D",color="white",markersize=6,zorder=6,markeredgecolor="#333",markeredgewidth=0.9)

    for i,medal in enumerate(["🥇","🥈","🥉"]):
        cx = i*(2*w+gap+0.20); mx = fin[fin["TeamName_norm"]==all_teams[i]]["Score"].max()
        ax_box.text(cx, mx+2.8, medal, ha="center", fontsize=15)

    xt = [i*(2*w+gap+0.20) for i in range(len(all_teams))]
    ax_box.set_xticks(xt); ax_box.set_xticklabels(labels, fontsize=FS_TICK+1)
    ax_box.set_ylabel("Score ↑", fontsize=FS_LABEL)
    ax_box.set_title("(a)  Score Distributions by Team & Track", fontsize=FS_TITLE, fontweight="bold", loc="left", color=GREY, pad=10)
    ax_box.set_xlim(-0.7, xt[-1]+0.7); style_ax(ax_box)
    ax_box.legend(handles=[
        mpatches.Patch(facecolor=PLAN_COL,alpha=0.82,label="Task Planning",edgecolor="#333"),
        mpatches.Patch(facecolor=EXEC_COL,alpha=0.82,label="Task Execution",edgecolor="#333"),
        Line2D([0],[0],marker="D",color="w",markerfacecolor="white",markeredgecolor="#333",markersize=7,label="Mean"),
    ], loc="upper right", fontsize=FS_ANNOT, frameon=True, framealpha=0.92)

    both      = [t for t in all_teams if t in best.index and "Task Planning" in best.columns and "Task Execution" in best.columns and not pd.isna(best.loc[t,"Task Planning"]) and not pd.isna(best.loc[t,"Task Execution"])]
    plan_only = [t for t in all_teams if t in best.index and "Task Planning" in best.columns and not pd.isna(best.loc[t].get("Task Planning",np.nan)) and ("Task Execution" not in best.columns or pd.isna(best.loc[t].get("Task Execution",np.nan)))]
    exec_only = [t for t in all_teams if t in best.index and "Task Execution" in best.columns and not pd.isna(best.loc[t].get("Task Execution",np.nan)) and ("Task Planning" not in best.columns or pd.isna(best.loc[t].get("Task Planning",np.nan)))]

    for t in both:
        px,py = best.loc[t,"Task Planning"], best.loc[t,"Task Execution"]
        ax_sc.scatter(px,py,s=220,color=TEAM_META[t][1],edgecolors="white",linewidths=1.4,zorder=5,alpha=0.93)
        ax_sc.annotate(TEAM_META[t][0],(px,py),textcoords="offset points",xytext=(7,4),fontsize=FS_ANNOT,color=GREY,fontweight="bold")
    for t in plan_only:
        px = best.loc[t].get("Task Planning",np.nan)
        if not pd.isna(px):
            ax_sc.scatter(px,-3,s=160,color=TEAM_META[t][1],marker="^",edgecolors="white",linewidths=1,zorder=5,alpha=0.88,clip_on=False)
            ax_sc.annotate(TEAM_META[t][0],(px,-3),textcoords="offset points",xytext=(4,-14),fontsize=FS_ANNOT-1,color=GREY,rotation=30)
    for t in exec_only:
        py = best.loc[t].get("Task Execution",np.nan)
        if not pd.isna(py):
            ax_sc.scatter(-3,py,s=160,color=TEAM_META[t][1],marker=">",edgecolors="white",linewidths=1,zorder=5,alpha=0.88,clip_on=False)
            ax_sc.annotate(TEAM_META[t][0],(-3,py),textcoords="offset points",xytext=(-52,3),fontsize=FS_ANNOT-1,color=GREY)

    ax_sc.plot([0,80],[0,80],"--",color="#AAAAAA",linewidth=1.2,alpha=0.7,zorder=1)
    ax_sc.text(70,73,"Equal",fontsize=FS_ANNOT-1,color="#AAAAAA",rotation=45)
    ax_sc.set_xlim(-8,80); ax_sc.set_ylim(-8,80)
    ax_sc.set_xlabel("Best Planning Score ↑",fontsize=FS_LABEL); ax_sc.set_ylabel("Best Execution Score ↑",fontsize=FS_LABEL)
    ax_sc.set_title("(b)  Planning vs. Execution Peak Scores\n(▲ = planning only,  ▶ = execution only)",fontsize=FS_TITLE,fontweight="bold",loc="left",color=GREY,pad=8)
    style_ax(ax_sc, xgrid=True)

    pc = [cnts.loc[t,"Task Planning"]  if (t in cnts.index and "Task Planning"  in cnts.columns) else 0 for t in all_teams]
    ec = [cnts.loc[t,"Task Execution"] if (t in cnts.index and "Task Execution" in cnts.columns) else 0 for t in all_teams]
    x  = np.arange(len(all_teams))
    ax_bar.bar(x,pc,color=PLAN_COL,alpha=0.82,label="Task Planning",edgecolor="white",linewidth=0.7)
    ax_bar.bar(x,ec,bottom=pc,color=EXEC_COL,alpha=0.82,label="Task Execution",edgecolor="white",linewidth=0.7)
    for i,(p,e) in enumerate(zip(pc,ec)):
        tot=p+e
        if tot>0: ax_bar.text(i,tot+0.4,str(tot),ha="center",va="bottom",fontsize=FS_TICK,color=GREY,fontweight="bold")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(labels,fontsize=FS_TICK+1)
    ax_bar.set_ylabel("Finished Submissions",fontsize=FS_LABEL)
    ax_bar.set_title("(c)  Submission Volume by Track",fontsize=FS_TITLE,fontweight="bold",loc="left",color=GREY,pad=10)
    ax_bar.legend(fontsize=FS_ANNOT,frameon=True,framealpha=0.92,loc="upper right"); style_ax(ax_bar)

    fig.suptitle("AssetOpsBench — Multi-Track Submission Analysis",fontsize=FS_SUPER,fontweight="bold",color=GREY,y=0.97)
    save_fig(fig,"fig1_overview")

# ── FIGURE 2 ──────────────────────────────────────────────────────────────────
def figure2():
    fig, axes = plt.subplots(1,2,figsize=(20,7),facecolor="white",gridspec_kw={"wspace":0.28})
    fig.subplots_adjust(left=0.07,right=0.97,top=0.88,bottom=0.13)
    for ax,track,panel in zip(axes,["Task Planning","Task Execution"],["(a)","(b)"]):
        sub = fin[fin["Track"]==track].sort_values("Date")
        ax.scatter(sub["Date"],sub["Score"],color="#DDDDDD",s=22,alpha=0.50,zorder=1)
        for t in all_teams:
            ts = sub[sub["TeamName_norm"]==t].sort_values("Date")
            if ts.empty: continue
            c,lb = TEAM_META[t][1], TEAM_META[t][0]
            ax.scatter(ts["Date"],ts["Score"],color=c,s=65,edgecolors="white",linewidths=0.8,zorder=4,alpha=0.92)
            ts = ts.copy(); ts["running_max"] = ts["Score"].cummax()
            ax.step(ts["Date"],ts["running_max"],color=c,linewidth=2.0,alpha=0.75,where="post",zorder=3)
            last = ts.iloc[-1]
            ax.annotate(lb,(last["Date"],last["running_max"]),textcoords="offset points",xytext=(6,3),fontsize=FS_ANNOT,color=c,fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(),rotation=30,ha="right",fontsize=FS_TICK)
        ax.set_ylabel("Score ↑",fontsize=FS_LABEL); ax.set_xlabel("Submission Date",fontsize=FS_LABEL)
        ax.set_title(f"{panel}  {track} — Score Trajectory\n(step line = running best per team)",fontsize=FS_TITLE,fontweight="bold",loc="left",color=GREY)
        style_ax(ax)
    fig.suptitle("AssetOpsBench — Score Progression over Competition Window",fontsize=FS_SUPER,fontweight="bold",color=GREY)
    save_fig(fig,"fig2_trajectory")

# ── FIGURE 3 — Heatmap only ───────────────────────────────────────────────────
def figure3():
    fig, ax = plt.subplots(figsize=(16,7), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.18)
    fin2 = fin.copy(); fin2["Week"] = fin2["Date"].dt.to_period("W").astype(str)
    weeks = sorted(fin2["Week"].unique())
    heat = np.zeros((len(all_teams),len(weeks)))
    for i,t in enumerate(all_teams):
        for j,w in enumerate(weeks):
            heat[i,j] = len(fin2[(fin2["TeamName_norm"]==t)&(fin2["Week"]==w)])
    cmap = LinearSegmentedColormap.from_list("ab",["#FFFFFF","#C6D9F1","#3A7DCC","#1A3A6C"],N=256)
    im = ax.imshow(heat,aspect="auto",cmap=cmap,interpolation="nearest",vmin=0)
    cbar = plt.colorbar(im,ax=ax,shrink=0.88,pad=0.02)
    cbar.set_label("Finished Submissions",fontsize=FS_LABEL); cbar.ax.tick_params(labelsize=FS_TICK)
    for i in range(len(all_teams)):
        for j in range(len(weeks)):
            v = int(heat[i,j])
            if v>0: ax.text(j,i,str(v),ha="center",va="center",fontsize=FS_TICK,fontweight="bold",color="white" if v>=5 else "#222")
    ax.set_xticks(range(len(weeks))); ax.set_xticklabels([w.split("/")[0][5:] for w in weeks],rotation=40,ha="right",fontsize=FS_TICK)
    ax.set_yticks(range(len(all_teams))); ax.set_yticklabels(labels,fontsize=FS_TICK+1)
    ax.set_xlabel("Week Starting",fontsize=FS_LABEL)
    ax.set_title("Submission Activity Heatmap — Weekly Finished Submissions per Team",fontsize=FS_TITLE+1,fontweight="bold",color=GREY,pad=12)
    ax.tick_params(axis="both",which="both",length=0)
    fig.suptitle("AssetOpsBench — Team Activity Patterns",fontsize=FS_SUPER,fontweight="bold",color=GREY,y=0.97)
    save_fig(fig,"fig3_heatmap")

# ── FIGURE 4 ──────────────────────────────────────────────────────────────────
def figure4():
    fig,(ax_imp,ax_suc) = plt.subplots(1,2,figsize=(18,7),facecolor="white",gridspec_kw={"wspace":0.30})
    fig.subplots_adjust(left=0.07,right=0.97,top=0.88,bottom=0.13)
    for t in all_teams:
        ts = fin[fin["TeamName_norm"]==t].sort_values("Date").reset_index(drop=True)
        if len(ts)<2: continue
        ts["cum_best"]=ts["Score"].cummax(); ts["sub_num"]=np.arange(1,len(ts)+1)
        ax_imp.plot(ts["sub_num"],ts["cum_best"],color=TEAM_META[t][1],linewidth=2.2,alpha=0.88,
                    marker="o",markersize=5,markeredgecolor="white",markeredgewidth=0.7,label=TEAM_META[t][0])
    ax_imp.set_xlabel("Submission Number (chronological)",fontsize=FS_LABEL)
    ax_imp.set_ylabel("Cumulative Best Score ↑",fontsize=FS_LABEL)
    ax_imp.set_title("(a)  Learning Curves — Cumulative Best Score",fontsize=FS_TITLE,fontweight="bold",loc="left",color=GREY)
    style_ax(ax_imp,xgrid=True); ax_imp.legend(fontsize=FS_TICK,ncol=2,frameon=True,framealpha=0.92,loc="lower right")

    all_df = df[df["TeamName_norm"].isin(all_teams)].copy(); x = np.arange(len(all_teams))
    for track,tc,mk in zip(["Task Planning","Task Execution"],[PLAN_COL,EXEC_COL],["o","s"]):
        rates=[]
        for t in all_teams:
            sub=all_df[(all_df["TeamName_norm"]==t)&(all_df["Track"]==track)]
            total=len(sub[sub["Status"].isin(["Finished","Failed","Cancelled"])]); done=len(sub[sub["Status"]=="Finished"])
            rates.append(done/total*100 if total>0 else np.nan)
        ax_suc.scatter(x,rates,color=tc,s=110,marker=mk,zorder=4,edgecolors="white",linewidths=0.9,label=track,alpha=0.92)
        ax_suc.plot(x,rates,color=tc,linewidth=1.0,alpha=0.30,zorder=2)
    ax_suc.set_xticks(range(len(all_teams))); ax_suc.set_xticklabels(labels,fontsize=FS_TICK+1)
    ax_suc.set_ylabel("Success Rate (%)",fontsize=FS_LABEL); ax_suc.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax_suc.set_ylim(0,112); ax_suc.axhline(100,color="#AAAAAA",linewidth=1.0,linestyle="--")
    ax_suc.set_title("(b)  Submission Success Rate by Track\n(Finished ÷ Total attempted)",fontsize=FS_TITLE,fontweight="bold",loc="left",color=GREY)
    style_ax(ax_suc); ax_suc.legend(fontsize=FS_ANNOT,frameon=True,framealpha=0.92)

    fig.suptitle("AssetOpsBench — Team Learning & Reliability",fontsize=FS_SUPER,fontweight="bold",color=GREY)
    save_fig(fig,"fig4_learning_success")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loaded {len(df)} rows · {fin['TeamName_norm'].nunique()} unique teams · output → {args.out}/")
    print("Generating Figure 1 …"); figure1()
    print("Generating Figure 2 …"); figure2()
    print("Generating Figure 3 …"); figure3()
    print("Generating Figure 4 …"); figure4()
    print("\nDone — all figures saved.")