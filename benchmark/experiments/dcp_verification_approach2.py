import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import anthropic
import re
import logging

logging.basicConfig(level=logging.INFO)

# ── Config ───────────────────────────────────────────────────────────────────
FILE1        = "diagnostic_step_to_id_mapping.csv"
FILE2        = "unique_tests.csv"
TOP_K        = 10
EMBED_MODEL  = "all-MiniLM-L6-v2"
CROSS_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_WEIGHT  = 0.3
EMBED_WEIGHT = 0.7
CANDIDATE_K  = 30      # candidates passed to cross-encoder
USE_LLM      = False   # set True to enable LLM reranking (uses Anthropic API)
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Load & Prepare Data ───────────────────────────────────────────────────
df1 = pd.read_csv(FILE1).dropna(subset=["diagnostic_steps"])
df2 = pd.read_csv(FILE2).dropna(subset=["Diagnostic Steps"])

df1["diagnostic_ids"] = df1["diagnostic_ids"].str.replace(r"[.?]+$", "", regex=True).str.strip()
id_to_text = dict(zip("D" + df2["Diagnostic ID"].astype(str), df2["Diagnostic Steps"]))

texts1 = df1["diagnostic_steps"].tolist()
texts2 = df2["Diagnostic Steps"].tolist()

# Enrich file2 with all available context
df2["combined_text"] = (
    "Diagnostic Steps: " + df2["Diagnostic Steps"].fillna("") + " | " +
    "Measurement Tools: " + df2["Measurement Tools"].fillna("")
)
texts2_enriched = df2["combined_text"].tolist()


# ── 2. BM25 Setup ────────────────────────────────────────────────────────────
def tokenize(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()

print("Building BM25 index...")
tokenized_corpus = [tokenize(t) for t in texts2_enriched]
bm25 = BM25Okapi(tokenized_corpus)


# ── 3. Embedding Setup ───────────────────────────────────────────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Encoding texts...")
embeddings1 = embed_model.encode(texts1,         convert_to_tensor=True, show_progress_bar=True)
embeddings2 = embed_model.encode(texts2_enriched, convert_to_tensor=True, show_progress_bar=True)


# ── 4. Cross-Encoder Setup ───────────────────────────────────────────────────
print("Loading cross-encoder...")
cross_encoder = CrossEncoder(CROSS_MODEL)


# ── 5. LLM Reranking (optional) ──────────────────────────────────────────────
def llm_rerank(query, candidates, top_k=TOP_K):
    """Use Claude to pick the best matching diagnostic step."""
    client = anthropic.Anthropic()
    candidate_str = "\n".join(
        [f"{i+1}. {c}" for i, c in enumerate(candidates)]
    )
    prompt = f"""You are a diagnostic matching expert.

Given the following diagnostic step from a case:
"{query}"

Rank these candidate diagnostic steps from best to worst match (return only the numbers in order, comma-separated):
{candidate_str}

Return ONLY the ranked numbers like: 2,1,4,3,5"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    ranked_str = response.content[0].text.strip()
    ranked_indices = [int(x.strip()) - 1 for x in ranked_str.split(",") if x.strip().isdigit()]
    return ranked_indices[:top_k]


# ── 6. Hybrid Retrieval + Reranking Pipeline ─────────────────────────────────
def normalize(scores):
    s = np.array(scores, dtype=float)
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


results = []
for i, (emb1, row1) in enumerate(zip(embeddings1, df1.itertuples())):

    # ── Step A: BM25 scores ──────────────────────────────────────────────────
    bm25_scores = np.array(bm25.get_scores(tokenize(row1.diagnostic_steps)))

    # ── Step B: Embedding scores ─────────────────────────────────────────────
    cos_scores = util.cos_sim(emb1, embeddings2)[0].cpu().numpy()

    # ── Step C: Hybrid fusion ────────────────────────────────────────────────
    hybrid_scores = (
        BM25_WEIGHT  * normalize(bm25_scores) +
        EMBED_WEIGHT * normalize(cos_scores)
    )
    # Get top CANDIDATE_K for cross-encoder reranking
    top_candidate_indices = np.argsort(hybrid_scores)[::-1][:CANDIDATE_K]

    # ── Step D: Cross-Encoder Reranking ─────────────────────────────────────
    pairs = [(row1.diagnostic_steps, texts2_enriched[j]) for j in top_candidate_indices]
    ce_scores = cross_encoder.predict(pairs)
    reranked_order = np.argsort(ce_scores)[::-1][:TOP_K]
    final_indices = [top_candidate_indices[r] for r in reranked_order]

    # ── Step E: Optional LLM Reranking ──────────────────────────────────────
    if USE_LLM:
        candidate_texts = [texts2[j] for j in final_indices]
        llm_order = llm_rerank(row1.diagnostic_steps, candidate_texts)
        final_indices = [final_indices[r] for r in llm_order if r < len(final_indices)]

    # ── Build result row ─────────────────────────────────────────────────────
    ground_truth_ids = set(gt.strip() for gt in str(row1.diagnostic_ids).split(","))
    row = {
        "case_id"          : row1.case_id,
        "diagnostic_steps" : row1.diagnostic_steps,
        "ground_truth_ids" : row1.diagnostic_ids,
        "ground_truth_text" : id_to_text.get(row1.diagnostic_ids.strip(), "NOT FOUND"),
    }
    predicted_ids = []
    for rank, idx in enumerate(final_indices, start=1):
        matched = df2.iloc[idx]
        pred_id = "D" + str(matched["Diagnostic ID"])
        row[f"top{rank}_diagnostic_id"]    = pred_id
        row[f"top{rank}_diagnostic_steps"] = matched["Diagnostic Steps"]
        row[f"top{rank}_score"]            = round(float(ce_scores[reranked_order[rank-1]]), 4)
        predicted_ids.append(pred_id)

    row["hit"] = any(gt in predicted_ids for gt in ground_truth_ids)
    row["status"] = "✅ hit" if row["hit"] else "❌ miss"   # ← add this
    results.append(row)


# ── 7. Accuracy Report ───────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
total   = len(results_df)
correct = results_df["hit"].sum()

print("\n" + "="*55)
print("  Full Pipeline Accuracy Report")
print("="*55)
print(f"  Embedding Model  : {EMBED_MODEL}")
print(f"  Cross-Encoder    : {CROSS_MODEL}")
print(f"  BM25 Weight      : {BM25_WEIGHT} | Embed Weight: {EMBED_WEIGHT}")
print(f"  LLM Reranking    : {'ON' if USE_LLM else 'OFF'}")
print(f"  Total Cases      : {total}")
print(f"  Correct          : {correct}")
print(f"  Accuracy         : {correct/total*100:.2f}%")
print("="*55)
for k in range(1, TOP_K + 1):
    topk_cols = [f"top{r}_diagnostic_id" for r in range(1, k + 1)]
    topk_hits = results_df.apply(
        lambda r: any(
            gt.strip() in [r[c] for c in topk_cols]
            for gt in str(r["ground_truth_ids"]).split(",")
        ), axis=1
    ).sum()
    print(f"  Top-{k} Accuracy: {topk_hits}/{total} = {topk_hits/total*100:.2f}%")
print("="*55)

results_df = results_df.sort_values("hit", ascending=True).reset_index(drop=True)
results_df.to_csv("matched_diagnostics_full_pipeline.csv", index=False)
print("\nResults saved to matched_diagnostics_full_pipeline.csv")