"""
failure_mode_analysis.py
Analysis on failure modes for the NeurIPS evaluation paper.
Input:  Best Submissions and collected trajectories of codabench competition
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

#!unzip /content/collected_files.zip -d /content/collected_files

import os

folder_path = "/content/collected_files"

image_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

image_count = 0
csv_count = 0

for file in os.listdir(folder_path):
    if file.lower().endswith(image_exts):
        image_count += 1
    elif file.lower().endswith('.csv'):
        csv_count += 1

print("Images:", image_count)
print("CSV files:", csv_count)

# ==============================
# Failure-mode extraction from sunburst images
# PaddleOCR with MKLDNN/PIR disabled
# ==============================

# IMPORTANT:
# Run this in a fresh Colab runtime.
# These environment variables must be set BEFORE importing paddle / paddleocr.

import os

# Workarounds for the oneDNN / PIR crash:
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

# Optional: reduce noisy logs
os.environ["GLOG_minloglevel"] = "2"

# --- installs (run once if needed) ---
# !pip install -q "paddleocr[all]" opencv-python pandas numpy scikit-learn sentence-transformers

import re
import math
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from paddleocr import PaddleOCR

# ==============================
# CONFIG
# ==============================
INPUT_DIR = "/content/collected_files"   # change this
OUTPUT_DIR = "/content/output_failure_modes"
EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

UPSCALE_FACTOR = 2
MIN_CONF = 0.35

CENTER_T = 0.18
INNER_T = 0.40
MIDDLE_T = 0.68

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# OCR INITIALIZATION
# ==============================
def build_ocr():
    """
    Try a few PaddleOCR 3.x-compatible configs.
    We keep MKLDNN disabled in the environment to avoid the crash.
    """
    candidates = [
        dict(lang="en", use_textline_orientation=True, enable_mkldnn=False),
        dict(lang="en", use_textline_orientation=True),
        dict(lang="en", enable_mkldnn=False),
        dict(lang="en"),
    ]

    last_error = None
    for kwargs in candidates:
        try:
            return PaddleOCR(**kwargs)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to initialize PaddleOCR. Last error: {last_error}")

ocr = build_ocr()

# ==============================
# HELPERS
# ==============================
def preprocess_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    h, w = img.shape[:2]
    img = cv2.resize(
        img,
        (int(w * UPSCALE_FACTOR), int(h * UPSCALE_FACTOR)),
        interpolation=cv2.INTER_CUBIC,
    )

    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return img


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("•·:;,.()[]{}<>|")
    return text


def is_percent_text(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}\s*%", clean_text(text)))


def percent_value(text: str):
    m = re.search(r"(\d{1,3})\s*%", clean_text(text))
    return int(m.group(1)) if m else None


def normalize_label(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\s*\d{1,3}\s*%\s*$", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z\u00C0-\u024F]", text))


def poly_center(poly) -> tuple[float, float]:
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def poly_area(poly) -> float:
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def ring_zone(x: float, y: float, w: int, h: int) -> tuple[str, float]:
    cx, cy = w / 2.0, h / 2.0
    r = math.hypot(x - cx, y - cy)
    max_r = math.hypot(cx, cy)
    ratio = r / max_r if max_r > 0 else 0.0

    if ratio < CENTER_T:
        zone = "center"
    elif ratio < INNER_T:
        zone = "inner_ring"
    elif ratio < MIDDLE_T:
        zone = "middle_ring"
    else:
        zone = "outer_ring"

    return zone, ratio


def safe_get_payload(res_obj):
    payload = getattr(res_obj, "json", None)
    if isinstance(payload, dict):
        if "res" in payload and isinstance(payload["res"], dict):
            return payload["res"]
        return payload
    if isinstance(res_obj, dict):
        if "res" in res_obj and isinstance(res_obj["res"], dict):
            return res_obj["res"]
        return res_obj
    raise TypeError(f"Unsupported PaddleOCR result type: {type(res_obj)}")


def run_predict(img):
    """
    PaddleOCR 3.x uses predict().
    We keep a tiny fallback for older wrappers.
    """
    try:
        return ocr.predict(img)
    except Exception:
        return ocr.ocr(img)


def extract_ocr_rows_for_image(image_path: str):
    img = preprocess_image(image_path)
    h, w = img.shape[:2]

    results = run_predict(img)
    rows = []

    if results is None:
        return rows, w, h

    # Common PaddleOCR 3.x return shape: iterable of result objects
    for res in results:
        try:
            payload = safe_get_payload(res)
        except Exception:
            continue

        rec_texts = payload.get("rec_texts", []) or []
        rec_scores = payload.get("rec_scores", []) or []
        rec_polys = payload.get("rec_polys", None)

        if rec_polys is None:
            rec_polys = payload.get("dt_polys", []) or []

        n = min(len(rec_texts), len(rec_scores), len(rec_polys))
        if n == 0:
            continue

        for i in range(n):
            raw = clean_text(rec_texts[i])
            score = float(rec_scores[i])

            if not raw or score < MIN_CONF:
                continue

            poly = rec_polys[i]
            cx, cy = poly_center(poly)
            area = poly_area(poly)
            zone, radius_ratio = ring_zone(cx, cy, w, h)

            rows.append({
                "image_file": os.path.basename(image_path),
                "image_path": image_path,
                "text_raw": raw,
                "text_clean": clean_text(raw),
                "text_norm": normalize_label(raw),
                "score": score,
                "x_center": cx,
                "y_center": cy,
                "bbox_area": area,
                "radius_ratio": radius_ratio,
                "ring_zone": zone,
                "is_percent": is_percent_text(raw),
                "percent_value": percent_value(raw),
            })

    return rows, w, h


def keep_label(row):
    if row["is_percent"]:
        return False
    t = clean_text(row["text_norm"])
    if len(t) < 3:
        return False
    if not has_letters(t):
        return False
    return True


# ==============================
# PROCESS FOLDER
# ==============================
image_files = sorted(
    str(p) for p in Path(INPUT_DIR).iterdir()
    if p.is_file() and p.suffix.lower() in EXTS
)

print(f"Found {len(image_files)} image files.")

all_rows = []

for idx, img_path in enumerate(image_files, 1):
    try:
        rows, w, h = extract_ocr_rows_for_image(img_path)
        all_rows.extend(rows)
        print(f"[{idx}/{len(image_files)}] OK  {os.path.basename(img_path)}  -> {len(rows)} OCR hits")
    except Exception as e:
        print(f"[{idx}/{len(image_files)}] ERR {os.path.basename(img_path)}  -> {e}")

df = pd.DataFrame(all_rows)

if df.empty:
    raise RuntimeError("No OCR text was extracted from any image.")

# Save raw OCR output
raw_csv = os.path.join(OUTPUT_DIR, "ocr_all_items.csv")
df.to_csv(raw_csv, index=False)

# ==============================
# FILTER + DEDUPE LABELS
# ==============================
df_labels = df[df.apply(keep_label, axis=1)].copy()

if not df_labels.empty:
    df_labels["x_bucket"] = (df_labels["x_center"] / 40).round().astype(int)
    df_labels["y_bucket"] = (df_labels["y_center"] / 40).round().astype(int)
    df_labels["dedupe_key"] = (
        df_labels["image_file"].astype(str) + "||" +
        df_labels["ring_zone"].astype(str) + "||" +
        df_labels["text_norm"].str.lower().astype(str) + "||" +
        df_labels["x_bucket"].astype(str) + "||" +
        df_labels["y_bucket"].astype(str)
    )
    df_labels = df_labels.drop_duplicates("dedupe_key").copy()

# ==============================
# AGGREGATIONS
# ==============================
label_counts = (
    df_labels["text_norm"]
    .str.strip()
    .value_counts()
    .reset_index()
)
label_counts.columns = ["label", "count"]

ring_counts = (
    df_labels.groupby(["ring_zone", "text_norm"])
    .size()
    .reset_index(name="count")
    .sort_values(["ring_zone", "count"], ascending=[True, False])
)

per_image_counts = (
    df_labels.groupby(["image_file", "text_norm"])
    .size()
    .reset_index(name="count")
    .sort_values(["image_file", "count"], ascending=[True, False])
)

pct_df = df[df["is_percent"]].copy()
pct_counts = (
    pct_df["percent_value"]
    .dropna()
    .astype(int)
    .value_counts()
    .sort_index()
    .reset_index()
)
pct_counts.columns = ["percent", "count"]

image_summary = (
    df_labels.groupby("image_file")
    .agg(
        extracted_labels=("text_norm", "nunique"),
        total_label_hits=("text_norm", "size"),
        mean_score=("score", "mean"),
    )
    .reset_index()
    .sort_values("total_label_hits", ascending=False)
)

# ==============================
# SAVE OUTPUTS
# ==============================
label_counts_csv = os.path.join(OUTPUT_DIR, "../../../../../Downloads/failure_mode_label_counts.csv")
ring_counts_csv = os.path.join(OUTPUT_DIR, "failure_mode_label_counts_by_ring.csv")
per_image_counts_csv = os.path.join(OUTPUT_DIR, "failure_mode_label_counts_per_image.csv")
pct_csv = os.path.join(OUTPUT_DIR, "percent_counts.csv")
image_summary_csv = os.path.join(OUTPUT_DIR, "image_summary.csv")

label_counts.to_csv(label_counts_csv, index=False)
ring_counts.to_csv(ring_counts_csv, index=False)
per_image_counts.to_csv(per_image_counts_csv, index=False)
pct_counts.to_csv(pct_csv, index=False)
image_summary.to_csv(image_summary_csv, index=False)

print("\nSaved files:")
print(raw_csv)
print(label_counts_csv)
print(ring_counts_csv)
print(per_image_counts_csv)
print(pct_csv)
print(image_summary_csv)

print("\nTop extracted failure modes:")
print(label_counts.head(25).to_string(index=False))

print("\nTop extracted percentages:")
if not pct_counts.empty:
    print(pct_counts.head(20).to_string(index=False))
else:
    print("No percentages detected.")

import os
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
CSV_DIR = "/content/collected_files"   # change this
OUTPUT_DIR = "/content/csv_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1) LOAD ALL CSV FILES
# =========================
csv_files = sorted([str(p) for p in Path(CSV_DIR).glob("*.csv")])

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {CSV_DIR}")

dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
        print(f"Loaded {f} -> {df.shape}")
    except Exception as e:
        print(f"Skipped {f}: {e}")

if not dfs:
    raise RuntimeError("No CSV files could be loaded.")

data = pd.concat(dfs, ignore_index=True)

# =========================
# 2) VALIDATE COLUMNS
# =========================
expected_cols = {"title", "description"}
missing = expected_cols - set(data.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}. Found columns: {list(data.columns)}")

data = data[["title", "description", "source_file"]].copy()

# =========================
# 3) CLEAN TEXT
# =========================
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x

data["title_clean"] = data["title"].apply(clean_text)
data["description_clean"] = data["description"].apply(clean_text)

# Drop blank rows
data = data[(data["title_clean"] != "") | (data["description_clean"] != "")].copy()

# =========================
# 4) BASIC FREQUENCY ANALYSIS
# =========================
title_counts = (
    data["title_clean"]
    .value_counts()
    .reset_index()
)
title_counts.columns = ["title", "count"]

title_counts.to_csv(os.path.join(OUTPUT_DIR, "../../../../../Downloads/title_counts.csv"), index=False)

desc_counts = (
    data["description_clean"]
    .value_counts()
    .reset_index()
)
desc_counts.columns = ["description", "count"]

desc_counts.to_csv(os.path.join(OUTPUT_DIR, "../../../../../Downloads/description_counts.csv"), index=False)

# =========================
# 5) NORMALIZE TITLES FOR GROUPING
# =========================
def normalize_title(t):
    t = clean_text(t).lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

data["title_norm"] = data["title_clean"].apply(normalize_title)

# Merge obvious duplicates after normalization
norm_title_counts = (
    data["title_norm"]
    .value_counts()
    .reset_index()
)
norm_title_counts.columns = ["title_norm", "count"]
norm_title_counts.to_csv(os.path.join(OUTPUT_DIR, "../../../../../Downloads/normalized_title_counts.csv"), index=False)

# =========================
# 6) COMMON PHRASES IN DESCRIPTIONS
# =========================
# This finds repeated 2- and 3-word phrases
desc_texts = data["description_clean"].tolist()

vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(2, 3),
    min_df=2
)

X = vectorizer.fit_transform(desc_texts)

phrase_counts = X.sum(axis=0).A1
phrases = vectorizer.get_feature_names_out()

phrase_df = pd.DataFrame({
    "phrase": phrases,
    "count": phrase_counts
}).sort_values("count", ascending=False)

phrase_df.to_csv(os.path.join(OUTPUT_DIR, "../../../../../Downloads/common_phrases.csv"), index=False)

# =========================
# 7) EMBEDDING-BASED CLUSTERING OF TITLES
# =========================
# This groups similar failure-mode titles even when wording differs.
unique_titles = data["title_clean"].dropna().drop_duplicates().tolist()

if len(unique_titles) >= 2:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(unique_titles, normalize_embeddings=True, show_progress_bar=True)

    # Cluster similarity threshold: lower = stricter clusters
    dist = cosine_distances(embeddings)
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=0.35
    )
    cluster_ids = clusterer.fit_predict(dist)

    cluster_map = pd.DataFrame({
        "title": unique_titles,
        "cluster_id": cluster_ids
    })

    # Add cluster IDs back to the full dataset
    data = data.merge(cluster_map, on="title", how="left")
    cluster_map.to_csv(os.path.join(OUTPUT_DIR, "../../../../../Downloads/title_clusters.csv"), index=False)
else:
    data["cluster_id"] = 0

# =========================
# 8) CLUSTER SUMMARY
# =========================
cluster_summary = (
    data.groupby("cluster_id")
    .agg(
        n_rows=("title", "size"),
        unique_titles=("title", "nunique"),
        sample_title=("title", "first"),
        sample_description=("description_clean", "first")
    )
    .reset_index()
    .sort_values("n_rows", ascending=False)
)

cluster_summary.to_csv(os.path.join(OUTPUT_DIR, "../../../../../Downloads/cluster_summary.csv"), index=False)

# =========================
# 9) TITLE + DESCRIPTION PAIRS
# =========================
pair_counts = (
    data.groupby(["title_clean", "description_clean"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
pair_counts.to_csv(os.path.join(OUTPUT_DIR, "../../../../../Downloads/title_description_pairs.csv"), index=False)

# =========================
# 10) QUICK TEXT STATS
# =========================
data["title_len"] = data["title_clean"].str.len()
data["desc_len"] = data["description_clean"].str.len()

stats = {
    "rows": len(data),
    "unique_titles": data["title_clean"].nunique(),
    "unique_descriptions": data["description_clean"].nunique(),
    "avg_title_len": float(data["title_len"].mean()),
    "avg_desc_len": float(data["desc_len"].mean())
}

pd.DataFrame([stats]).to_csv(os.path.join(OUTPUT_DIR, "basic_stats.csv"), index=False)

# =========================
# 11) PRINT TOP RESULTS
# =========================
print("\n=== BASIC STATS ===")
print(stats)

print("\n=== TOP TITLES ===")
print(title_counts.head(20).to_string(index=False))

print("\n=== TOP PHRASES IN DESCRIPTIONS ===")
print(phrase_df.head(30).to_string(index=False))

print("\n=== TOP CLUSTERS ===")
print(cluster_summary.head(20).to_string(index=False))

print(f"\nSaved outputs to: {OUTPUT_DIR}")

import os
import re
from pathlib import Path
from collections import Counter

import pandas as pd

# ============================================================
# CONFIG
# ============================================================
CSV_DIR = "/content/collected_files"                 # folder with your original CSVs
OUTPUT_DIR = "/content/csv_analysis_output"    # folder where title_clusters.csv already exists
CLUSTERS_FILE = os.path.join(OUTPUT_DIR, "../../../../../Downloads/title_clusters.csv")
TAXONOMY_FILE = os.path.join(OUTPUT_DIR, "../../../../../Downloads/failure_mode_taxonomy.csv")
REVIEW_FILE = os.path.join(OUTPUT_DIR, "../../../../../Downloads/taxonomy_review_needed.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
csv_files = sorted([str(p) for p in Path(CSV_DIR).glob("*.csv")])
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {CSV_DIR}")

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    df["source_file"] = os.path.basename(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

required = {"title", "description"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Found: {list(data.columns)}")

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x

data["title_clean"] = data["title"].apply(clean_text)
data["description_clean"] = data["description"].apply(clean_text)
data = data[(data["title_clean"] != "") | (data["description_clean"] != "")].copy()

# ============================================================
# ATTACH CLUSTER IDS
# ============================================================
if os.path.exists(CLUSTERS_FILE):
    clusters = pd.read_csv(CLUSTERS_FILE)
    if "title" not in clusters.columns or "cluster_id" not in clusters.columns:
        raise ValueError("title_clusters.csv must contain columns: title, cluster_id")

    clusters["title"] = clusters["title"].apply(clean_text)
    data = data.merge(
        clusters[["title", "cluster_id"]],
        left_on="title_clean",
        right_on="title",
        how="left"
    )
    data.drop(columns=["title"], inplace=True, errors="ignore")
else:
    raise FileNotFoundError(
        f"Could not find {CLUSTERS_FILE}. Run the clustering step first."
    )

# If some rows did not match, keep them but mark as unclustered
data["cluster_id"] = data["cluster_id"].fillna(-1).astype(int)

# ============================================================
# PARENT CATEGORY RULES
# Edit these if you want a different taxonomy.
# The order matters: first match wins.
# ============================================================
PARENT_RULES = [
    (
        "Answer Completion / Task Closure",
        [
            r"final answer",
            r"progress towards final answer",
            r"lack of final answer",
            r"no final answer",
            r"task completion",
            r"towards completion",
        ],
        "final_answer"
    ),
    (
        "Redundancy / Repetition",
        [
            r"redundant",
            r"repetitive",
            r"repeat",
            r"same query",
            r"same retrieval",
            r"without new insights",
            r"duplicate",
        ],
        "repetition"
    ),
    (
        "Context / Memory",
        [
            r"contextual understanding",
            r"lack of context",
            r"context",
            r"memory",
            r"context awareness",
        ],
        "context"
    ),
    (
        "Information Retrieval Strategy",
        [
            r"proactive information retrieval",
            r"information retrieval",
            r"retrieval",
            r"search",
            r"lookup",
        ],
        "retrieval"
    ),
    (
        "Data Availability / Access",
        [
            r"data unavail",
            r"unavailable",
            r"missing data",
            r"no data",
            r"data access",
        ],
        "data_access"
    ),
    (
        "Error Handling / Robustness",
        [
            r"error handling",
            r"inadequate error handling",
            r"incorrect file path",
            r"file path",
            r"exception",
            r"failed to handle",
        ],
        "error_handling"
    ),
    (
        "Entity / Sensor Consistency",
        [
            r"inconsistent sensor",
            r"inconsistent equipment",
            r"equipment id",
            r"sensor data",
            r"entity mismatch",
            r"identifier mismatch",
        ],
        "consistency"
    ),
    (
        "Task Progression / Planning",
        [
            r"task progression",
            r"lack of task progress",
            r"progression",
            r"planning",
            r"step progression",
        ],
        "planning"
    ),
    (
        "Execution / Action Redundancy",
        [
            r"redundant task execution",
            r"task execution",
            r"execution",
            r"repeated action",
        ],
        "execution"
    ),
]

def assign_parent(text: str):
    """
    Return (parent_category, theme_tag, matched_rule).
    """
    t = (text or "").lower()

    for parent, patterns, theme_tag in PARENT_RULES:
        for pat in patterns:
            if re.search(pat, t):
                return parent, theme_tag, pat

    return "Other / Review", "other", None

# ============================================================
# BUILD CLUSTER-LEVEL TAXONOMY
# ============================================================
rows = []

for cluster_id, g in data.groupby("cluster_id"):
    # cluster text to inspect
    titles = [t for t in g["title_clean"].dropna().tolist() if t]
    descs = [d for d in g["description_clean"].dropna().tolist() if d]

    # representative title = most frequent title in cluster
    title_counts = Counter(titles)
    representative_title = title_counts.most_common(1)[0][0] if title_counts else ""

    # representative description = first non-empty description
    representative_description = next((d for d in descs if d), "")

    # combined text for parent assignment
    cluster_text = " ".join(titles + descs)
    parent_category, theme_tag, matched_rule = assign_parent(cluster_text)

    # cluster subcategory is the cluster's own representative title
    subcategory_name = representative_title if representative_title else f"cluster_{cluster_id}"

    rows.append({
        "cluster_id": cluster_id,
        "parent_category": parent_category,
        "theme_tag": theme_tag,
        "subcategory_name": subcategory_name,
        "matched_rule": matched_rule if matched_rule else "",
        "n_rows": len(g),
        "unique_titles": g["title_clean"].nunique(),
        "representative_title": representative_title,
        "representative_description": representative_description,
        "all_titles": " | ".join(sorted(set(titles))[:20]),
    })

taxonomy = pd.DataFrame(rows)

# Sort: known parents first, review last
taxonomy["parent_sort"] = taxonomy["parent_category"].eq("Other / Review").astype(int)
taxonomy = taxonomy.sort_values(["parent_sort", "parent_category", "n_rows"], ascending=[True, True, False]).drop(columns=["parent_sort"])

# ============================================================
# REVIEW FILE FOR AMBIGUOUS CLUSTERS
# ============================================================
review = taxonomy[taxonomy["parent_category"] == "Other / Review"].copy()

# Save outputs
taxonomy.to_csv(TAXONOMY_FILE, index=False)
review.to_csv(REVIEW_FILE, index=False)

# ============================================================
# PRINT SUMMARY
# ============================================================
print(f"Saved taxonomy table to: {TAXONOMY_FILE}")
print(f"Saved review-needed clusters to: {REVIEW_FILE}")

print("\nParent category counts:")
print(taxonomy["parent_category"].value_counts().to_string())

print("\nTop taxonomy rows:")
print(taxonomy.head(25).to_string(index=False))

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Optional: for nicer hierarchical charts
import plotly.express as px

# =========================
# CONFIG
# =========================
OUTPUT_DIR = "/content/csv_analysis_output"
TAXONOMY_FILE = os.path.join(OUTPUT_DIR, "../../../../../Downloads/failure_mode_taxonomy.csv")
LABEL_COUNTS_FILE = os.path.join(OUTPUT_DIR, "../../../../../Downloads/failure_mode_label_counts.csv")
CLUSTERS_FILE = os.path.join(OUTPUT_DIR, "../../../../../Downloads/title_clusters.csv")

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================
# LOAD FILES
# =========================
taxonomy = pd.read_csv(TAXONOMY_FILE)
labels = pd.read_csv(LABEL_COUNTS_FILE)
clusters = pd.read_csv(CLUSTERS_FILE)

# Basic cleanup
for df in [taxonomy, labels, clusters]:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

# =========================
# 1) PARENT CATEGORY COUNTS
# =========================
parent_counts = (
    taxonomy.groupby("parent_category")["n_rows"]
    .sum()
    .sort_values(ascending=True)
)

plt.figure(figsize=(12, 7))
parent_counts.plot(kind="barh")
plt.xlabel("Number of rows")
plt.ylabel("Parent category")
plt.title("Failure Modes by Parent Category")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "../../../../../Downloads/parent_category_counts.png"), dpi=300)
plt.close()

# =========================
# 2) TOP SUBCATEGORIES
# =========================
sub_counts = (
    taxonomy.groupby(["parent_category", "subcategory_name"])["n_rows"]
    .sum()
    .reset_index()
    .sort_values("n_rows", ascending=False)
)

top_sub = sub_counts.head(20).sort_values("n_rows", ascending=True)

plt.figure(figsize=(14, 8))
plt.barh(top_sub["subcategory_name"], top_sub["n_rows"])
plt.xlabel("Number of rows")
plt.ylabel("Subcategory")
plt.title("Top 20 Failure-Mode Subcategories")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "../../../../../Downloads/top_subcategories.png"), dpi=300)
plt.close()

# =========================
# 3) CLUSTER SIZE DISTRIBUTION
# =========================
cluster_sizes = clusters["cluster_id"].value_counts().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.hist(cluster_sizes.values, bins=20)
plt.xlabel("Cluster size")
plt.ylabel("Number of clusters")
plt.title("Distribution of Cluster Sizes")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "../../../../../Downloads/cluster_size_distribution.png"), dpi=300)
plt.close()

# =========================
# 4) TOP LABELS ACROSS ALL IMAGES
# =========================
top_labels = labels.head(25).sort_values("count", ascending=True)

plt.figure(figsize=(14, 8))
plt.barh(top_labels["label"], top_labels["count"])
plt.xlabel("Count")
plt.ylabel("Label")
plt.title("Top 25 Failure-Mode Labels")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "../../../../../Downloads/top_labels.png"), dpi=300)
plt.close()

# =========================
# 5) TAXONOMY TREEMAP
# =========================
# Nice hierarchical view: parent -> subcategory
tree_df = taxonomy.copy()
tree_df = tree_df[tree_df["parent_category"].notna() & tree_df["subcategory_name"].notna()].copy()

fig = px.treemap(
    tree_df,
    path=["parent_category", "subcategory_name"],
    values="n_rows",
    color="parent_category",
    title="Failure-Mode Taxonomy Treemap"
)
fig.write_html(os.path.join(PLOT_DIR, "../../../../../Downloads/taxonomy_treemap.html"))

# =========================
# 6) TAXONOMY SUNBURST
# =========================
fig2 = px.sunburst(
    tree_df,
    path=["parent_category", "subcategory_name"],
    values="n_rows",
    title="Failure-Mode Taxonomy Sunburst"
)
fig2.write_html(os.path.join(PLOT_DIR, "../../../../../Downloads/taxonomy_sunburst.html"))

# =========================
# 7) PRINT SUMMARY
# =========================
print("Saved plots to:", PLOT_DIR)
print("PNG files:")
for f in sorted(Path(PLOT_DIR).glob("*.png")):
    print(" -", f.name)
print("HTML files:")
for f in sorted(Path(PLOT_DIR).glob("*.html")):
    print(" -", f.name)

import pandas as pd
import os
from pathlib import Path

CSV_DIR = "/content/collected_files"
OUTPUT_DIR = "/content/csv_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dfs = []
for f in Path(CSV_DIR).glob("*.csv"):
    df = pd.read_csv(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Clean titles
data["title"] = data["title"].astype(str).str.strip()

# Count labels
label_counts = (
    data["title"]
    .value_counts()
    .reset_index()
)
label_counts.columns = ["label", "count"]

# Save
out_path = os.path.join(OUTPUT_DIR, "../../../../../Downloads/failure_mode_label_counts.csv")
label_counts.to_csv(out_path, index=False)

print("Saved:", out_path)