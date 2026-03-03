from failure_mode_reduction import failure_mode_reduction

result = failure_mode_reduction(
    combined_pickle_path="./processed_trajectories/combined_m12_db.pkl",
    out_dir="summary_codabench",
    # model_name="all-MiniLM-L6-v2",  # or another sentence-transformers model
    # k=6,                             # fix cluster count if you prefer
)

print(result["k"], result["silhouette_scores"][:3])
print(result["paths"])
print(result["df_clustered"].head())
