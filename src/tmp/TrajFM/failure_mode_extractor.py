import argparse
from failure_mode_generator import process_trajectories
from failure_mode_reduction import failure_mode_reduction


def main():
    """
    Run the failure mode pipeline:
      1) Generate combined pickle from trajectories
      2) Reduce/cluster additional failure modes and export CSVs
    """
    parser = argparse.ArgumentParser(
        description="Analyze LLM execution trajectories to identify and cluster failure modes."
    )
    parser.add_argument(
        "--traj_directory",
        type=str,
        default="./localtemp/trajectory/",
        help="Path to the root directory containing per-timestamp trajectory folders.",
    )
    parser.add_argument(
        "--backstage_directory",
        type=str,
        default=".",
        help="(Optional) Path to auxiliary resources (unused, kept for compatibility).",
    )
    parser.add_argument(
        "--model_id",
        type=int,
        default=18,
        help="Model ID passed to the generator step.",
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default="summary",
        help="Directory to write the clustered CSV outputs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model for title embeddings.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Optional fixed number of clusters (if omitted, silhouette chooses K).",
    )
    parser.add_argument(
        "--timestamps",
        nargs="*",
        default=None,
        help="Optional list of timestamps to process. If omitted, auto-discovers all subfolders.",
    )

    args = parser.parse_args()

    # Step 1: Generate combined pickle (auto-discovers timestamps if not provided)
    gen = process_trajectories(
        timestamps=args.timestamps,  # None => auto-discover
        traj_root_base=args.traj_directory,
        model_id=args.model_id,
        out_dir="processed_trajectories",
    )
    print("\n[Step 1] Combined pickle:", gen["combined_path"])
    print(gen["combined_df"].head())

    # Step 2: Reduce/cluster additional failure modes from the combined pickle
    red = failure_mode_reduction(
        combined_pickle_path=gen["combined_path"],
        out_dir=args.summary_dir,
        model_name=args.model_name,
        k=args.k,
    )
    print("\n[Step 2] Chosen K:", red["k"])
    if red.get("silhouette_scores"):
        print("[Step 2] Silhouette scores (first 3):", red["silhouette_scores"][:3])
    print("[Step 2] Outputs:", red["paths"])
    print(red["df_clustered"].head())


if __name__ == "__main__":
    main()
