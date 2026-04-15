"""Failure mode reduction through clustering.

This module handles clustering of additional failure modes using sentence embeddings
and K-means clustering to group similar failure patterns.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List


def failure_mode_reduction(
    combined_pickle_path: str,
    out_dir: str = "./src/trajectory_analysis/failure_mode/results/summary",
    model_name: str = "all-MiniLM-L6-v2",
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 7,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Reduce additional failure modes by exploding, embedding, clustering, and labeling.

    This function processes additional failure modes from trajectory analysis by:
    1. Exploding the nested failure mode lists into individual records
    2. Embedding failure mode titles using sentence transformers
    3. Clustering similar failure modes using K-means
    4. Selecting representative titles for each cluster
    5. Exporting results to CSV files in results/summary/

    Args:
        combined_pickle_path: Path to the combined pickle file from trajectory processing
        out_dir: Output directory for CSV files (default: results/summary)
        model_name: Sentence transformer model name for embeddings (default: all-MiniLM-L6-v2)
        k: Fixed number of clusters (None = auto-select using silhouette score)
        k_min: Minimum K to consider when auto-selecting (default: 2)
        k_max: Maximum K to consider when auto-selecting (default: 7)
        verbose: Print progress messages (default: True)

    Returns:
        Dictionary containing:
            - df_expanded: DataFrame with exploded failure modes [title, description]
            - df_clustered: DataFrame with clusters [cluster, failure mode, title, description]
            - n_clusters: Number of clusters used
            - silhouette_scores: List of (k, score) tuples if auto-selected, else []
            - paths: Dict with CSV file paths

    Raises:
        KeyError: If required columns are missing from the pickle file

    Example:
        >>> results = failure_mode_reduction(
        ...     combined_pickle_path="./results/combined_m18_db.pkl",
        ...     out_dir="./results/summary",
        ...     k=5  # Fixed 5 clusters
        ... )
        >>> print(f"Created {results['k']} clusters")
        >>> print(results['df_clustered'].head())
    """
    if verbose:
        print(f"Loading combined pickle: {combined_pickle_path}")
    df = pd.read_pickle(combined_pickle_path)
    if verbose:
        print(f"Loaded DataFrame with {len(df)} rows")

    # --- Step 1: Explode addi_fm_list -> title/description ---
    if verbose:
        print("Exploding additional failure modes...")
    if "addi_fm_cnt" not in df.columns or "addi_fm_list" not in df.columns:
        raise KeyError("Expected columns 'addi_fm_cnt' and 'addi_fm_list' not found.")

    # Keep trajectory_path to link failures back to their source
    cols_to_keep = ["addi_fm_cnt", "addi_fm_list"]
    if "trajectory_path" in df.columns:
        cols_to_keep.append("trajectory_path")

    df_new_fm = df[df["addi_fm_cnt"] > 0][cols_to_keep].copy()
    df_new_fm.reset_index(drop=True, inplace=True)

    df_exploded = df_new_fm.explode("addi_fm_list", ignore_index=True)
    df_expanded = pd.concat(
        [
            df_exploded.drop(columns=["addi_fm_list"]),
            pd.json_normalize(df_exploded["addi_fm_list"]),
        ],
        axis=1,
    )

    keep_cols = [
        c
        for c in ["trajectory_path", "title", "description"]
        if c in df_expanded.columns
    ]
    if "title" not in keep_cols and "description" not in keep_cols:
        raise KeyError(
            "No 'title'/'description' columns found inside 'addi_fm_list' items."
        )
    df_expanded = df_expanded[keep_cols].copy()

    # Save the additional failure modes CSV
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    additional_csv = out / "additional_fm.csv"
    df_expanded.to_csv(additional_csv, index=False)
    if verbose:
        print(f"Saved: {additional_csv} (rows={len(df_expanded)})")

    # --- Step 2: Embeddings + clustering with small-sample handling ---
    titles = df_expanded["title"].fillna("").astype(str).tolist()
    n = len(titles)

    # n == 0: nothing to do
    if n == 0:
        if verbose:
            print("No titles to cluster. Returning early.")
        return {
            "df_expanded": df_expanded,
            "df_clustered": pd.DataFrame(
                columns=["cluster", "failure mode", "title", "description"]
            ),
            "k": 0,
            "silhouette_scores": [],
            "paths": {
                "additional_fm_csv": str(additional_csv),
                "additional_fm_clustered_csv": None,
            },
        }

    # n == 1: assign a single cluster without embeddings
    if n == 1:
        df_clustered = df_expanded.copy()
        df_clustered["cluster"] = 0
        df_clustered["failure mode"] = df_clustered["title"]
        clustered_csv = out / "additional_fm_clustered.csv"
        # Include trajectory_path if available
        cols = ["cluster", "failure mode"]
        if "trajectory_path" in df_clustered.columns:
            cols.append("trajectory_path")
        cols.extend(["title", "description"])
        df_clustered[cols].to_csv(clustered_csv, index=False)
        if verbose:
            print(f"Single item: saved {clustered_csv}")
        return {
            "df_expanded": df_expanded,
            "df_clustered": df_clustered[
                ["cluster", "failure mode", "title", "description"]
            ],
            "k": 1,
            "silhouette_scores": [],
            "paths": {
                "additional_fm_csv": str(additional_csv),
                "additional_fm_clustered_csv": str(clustered_csv),
            },
        }

    # n >= 2: embed and cluster
    if verbose:
        print(f"Embedding {n} titles with {model_name} ...")
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import euclidean_distances
    import numpy as np

    model = SentenceTransformer(model_name)
    embeddings = model.encode(titles, convert_to_numpy=True, show_progress_bar=False)

    silhouette_scores: List[Tuple[int, float]] = []

    # n == 2: only valid K is 2 for silhouette constraints
    if n == 2:
        k = 2
        if verbose:
            print("Only two samples detected; using K=2.")
    else:
        if k is None:
            lo = max(2, k_min)
            hi = min(k_max, n - 1)  # silhouette requires k <= n-1
            if lo > hi:
                # Not enough samples for a range; fall back to a valid K
                k = min(2, n - 1)
                if verbose:
                    print(f"Insufficient samples for a K range; using K={k}.")
            else:
                if verbose:
                    print(f"Selecting K by silhouette over [{lo}..{hi}]")
                best_k, best_score = None, -1.0
                for cand in range(lo, hi + 1):
                    km = KMeans(n_clusters=cand, random_state=42, n_init="auto")
                    labels = km.fit_predict(embeddings)
                    # If all points fall into one cluster (identical embeddings), silhouette is invalid
                    if len(set(labels)) <= 1:
                        score = -1.0
                    else:
                        score = float(silhouette_score(embeddings, labels))
                    silhouette_scores.append((cand, score))
                    if score > best_score:
                        best_k, best_score = cand, score
                k = best_k or min(2, n - 1)
                if verbose:
                    print("Silhouette scores:", silhouette_scores)
                    print(f"Chosen K = {k}")
        else:
            # user-provided K → clamp safely
            if n <= 2:
                k = 2
            else:
                k = max(2, min(int(k), n - 1))
            if verbose:
                print(f"Using K = {k} (validated for n={n})")

    # Final clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(embeddings)

    df_clustered = df_expanded.copy()
    df_clustered["cluster"] = clusters

    # Representative (closest to centroid) title per cluster
    if verbose:
        print("Selecting representative title for each cluster...")
    representative_titles: List[Tuple[int, str]] = []
    for cl in range(k):
        idxs = df_clustered.index[df_clustered["cluster"] == cl].tolist()
        if not idxs:
            continue
        dists = euclidean_distances(
            embeddings[idxs], [kmeans.cluster_centers_[cl]]
        ).flatten()
        closest_local = int(np.argmin(dists))
        rep_idx = idxs[closest_local]
        representative_titles.append((cl, df_clustered.loc[rep_idx, "title"]))

    if verbose and representative_titles:
        print("\nRepresentative titles:")
        for cl, title in representative_titles:
            print(f"  Cluster {cl}: {title}")

    cluster_to_title = dict(representative_titles)
    df_clustered["failure mode"] = df_clustered["cluster"].map(cluster_to_title)

    # Final column order - include trajectory_path if available
    cols = ["cluster", "failure mode"]
    if "trajectory_path" in df_clustered.columns:
        cols.append("trajectory_path")
    cols.extend(["title", "description"])
    df_clustered = df_clustered[cols].copy()

    clustered_csv = out / "additional_fm_clustered.csv"
    df_clustered.to_csv(clustered_csv, index=False)
    if verbose:
        print(f"Saved: {clustered_csv} (rows={len(df_clustered)})")

    return {
        "df_expanded": df_expanded,
        "df_clustered": df_clustered,
        "k": k,
        "silhouette_scores": silhouette_scores,
        "paths": {
            "additional_fm_csv": str(additional_csv),
            "additional_fm_clustered_csv": str(clustered_csv),
        },
    }


# Made with Bob
