import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# ── Config ──────────────────────────────────────────────────────────────────
FILE1 = "diagnostic_step_to_id_mapping.csv"
FILE2 = "unique_tests.csv"
TOP_K = 10
MODEL_NAME = "all-MiniLM-L6-v2"
# ────────────────────────────────────────────────────────────────────────────

# Load data
df1 = pd.read_csv(FILE1)
df2 = pd.read_csv(FILE2)

id_to_text = dict(zip("D" + df2["Diagnostic ID"].astype(str), df2["Diagnostic Steps"]))
df1["diagnostic_ids"] = df1["diagnostic_ids"].str.replace(r"[.?]+$", "", regex=True).str.strip()

# Drop rows where the text columns are null
df1 = df1.dropna(subset=["diagnostic_steps"])
df2 = df2.dropna(subset=["Diagnostic Steps"])


texts1 = df1["diagnostic_steps"].tolist()
texts2 = df2["Diagnostic Steps"].tolist()

# Load model
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

# Encode both columns
print("Encoding texts...")
embeddings1 = model.encode(texts1, convert_to_tensor=True, show_progress_bar=True)
embeddings2 = model.encode(texts2, convert_to_tensor=True, show_progress_bar=True)

# Find top-K matches and evaluate against ground truth
results = []
for i, (emb1, row1) in enumerate(zip(embeddings1, df1.itertuples())):
    cos_scores = util.cos_sim(emb1, embeddings2)[0]
    top_k_results = torch.topk(cos_scores, k=min(TOP_K, len(texts2)))

    # Ground truth IDs — handle multiple comma-separated IDs e.g. "D1,D3"
    ground_truth_ids = set(
        gt.strip() for gt in str(row1.diagnostic_ids).split(",")
    )

    row = {
        "case_id"          : row1.case_id,
        "diagnostic_steps" : row1.diagnostic_steps,
        "ground_truth_ids" : row1.diagnostic_ids,
        "ground_truth_text" : id_to_text.get(row1.diagnostic_ids.strip(), "NOT FOUND"),
    }

    predicted_ids = []
    for rank, (score, idx) in enumerate(zip(top_k_results.values, top_k_results.indices), start=1):
        matched_row = df2.iloc[idx.item()]
        pred_id = "D" + str(matched_row["Diagnostic ID"])
        row[f"top{rank}_diagnostic_id"]    = pred_id
        row[f"top{rank}_diagnostic_steps"] = matched_row["Diagnostic Steps"]
        row[f"top{rank}_score"]            = round(score.item(), 4)
        predicted_ids.append(pred_id)

    # Hit = any ground truth ID appears in the top-K predictions
    hit = any(gt in predicted_ids for gt in ground_truth_ids)
    row["hit"]    = hit
    row["status"] = "✅ hit" if hit else "❌ miss"   # ← easy to sort/filter

    results.append(row)

# ── Accuracy Report ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)

# ── Sort: misses first for easy inspection ───────────────────────────────────
results_df = results_df.sort_values("hit", ascending=True).reset_index(drop=True)

total   = len(results_df)
correct = results_df["hit"].sum()
accuracy = correct / total * 100

print("\n" + "="*50)
print(f"  Embedding Mapping Accuracy Report")
print("="*50)
print(f"  Model       : {MODEL_NAME}")
print(f"  Top-K       : {TOP_K}")
print(f"  Total Cases : {total}")
print(f"  Correct     : {correct}")
print(f"  Accuracy    : {accuracy:.2f}%")
print("="*50)

# Per top-K breakdown — how many were found at rank 1, 2, 3
for k in range(1, TOP_K + 1):
    topk_cols = [f"top{r}_diagnostic_id" for r in range(1, k + 1)]
    topk_hits = results_df.apply(
        lambda r: any(
            gt.strip() in [r[c] for c in topk_cols]
            for gt in str(r["ground_truth_ids"]).split(",")
        ), axis=1
    ).sum()
    print(f"  Top-{k} Accuracy: {topk_hits}/{total} = {topk_hits/total*100:.2f}%")

print("="*50)

# Save results — misses at top for easy inspection
results_df.to_csv("matched_diagnostics.csv", index=False)
print(f"\nResults saved to matched_diagnostics.csv")
print(f"  ❌ Misses : {total - correct} (sorted to top)")
print(f"  ✅ Hits   : {correct}")
print(results_df.head())