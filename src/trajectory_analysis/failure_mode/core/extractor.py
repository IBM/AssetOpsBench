"""Command-line interface for failure mode extraction and analysis.

This module provides a CLI for running the complete failure mode pipeline:
1. Generate failure modes from trajectories
2. Cluster similar failure modes
3. Export results to CSV files
"""

import argparse
import sys
from typing import Optional, List

from .generator import process_trajectories
from .reducer import failure_mode_reduction


def run_extraction_pipeline(
    traj_directory: str = ".",
    summary_dir: str = "./results/summary",
    model_name: str = "all-MiniLM-L6-v2",
    k: Optional[int] = None,
    timestamps: Optional[List[str]] = None,
    llm_backend=None,
    temperature: float = 0.0,
    verbose: bool = True,
):
    """
    Run the complete failure mode extraction pipeline.

    This function orchestrates the two-step process:
    1. Generate failure modes from trajectories using LLM analysis
    2. Cluster additional failure modes using sentence embeddings

    Args:
        traj_directory: Root directory containing trajectory files
        summary_dir: Output directory for clustered CSV files
        model_name: Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)
        k: Fixed number of clusters (None = auto-select using silhouette)
        timestamps: List of timestamps to process (None = auto-discover)
        llm_backend: LLM backend for generation (None = default Claude 4 Sonnet)
        temperature: LLM temperature parameter (default: 0.0)
        verbose: Print progress messages (default: True)

    Returns:
        Dictionary containing:
            - generation: Results from trajectory processing
            - reduction: Results from clustering

    Example:
        >>> from src.llm.litellm import LiteLLMBackend
        >>> llm = LiteLLMBackend("litellm_proxy/GCP/claude-4-sonnet")
        >>> results = run_extraction_pipeline(
        ...     traj_directory="./sample_trajectories",
        ...     llm_backend=llm,
        ...     k=5
        ... )
        >>> print(f"Processed {len(results['generation']['combined_df'])} trajectories")
        >>> print(f"Created {results['reduction']['k']} clusters")
    """
    if verbose:
        print("=" * 60)
        print("Failure Mode Extraction Pipeline")
        print("=" * 60)

    # Step 1: Generate failure modes from trajectories
    if verbose:
        print("\n[Step 1] Generating failure modes from trajectories...")
    gen = process_trajectories(
        timestamps=timestamps,
        traj_root_base=traj_directory,
        llm_backend=llm_backend,
        temperature=temperature,
        out_dir="./results",
    )
    if verbose:
        print(f"[Step 1] Combined pickle: {gen['combined_path']}")
        print(f"[Step 1] Processed {len(gen['combined_df'])} trajectories")

    # Step 2: Cluster additional failure modes
    if verbose:
        print("\n[Step 2] Clustering additional failure modes...")
    red = failure_mode_reduction(
        combined_pickle_path=gen["combined_path"],
        out_dir=summary_dir,
        model_name=model_name,
        k=k,
        verbose=verbose,
    )
    if verbose:
        print(f"[Step 2] Created {red['k']} clusters")
        if red.get("silhouette_scores"):
            print(f"[Step 2] Silhouette scores: {red['silhouette_scores'][:3]}...")
        print(f"[Step 2] Outputs: {red['paths']}")

    if verbose:
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)

    return {
        "generation": gen,
        "reduction": red,
    }


def main():
    """
    Command-line interface for failure mode extraction.

    This is the main entry point when running the script from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Analyze LLM execution trajectories to identify and cluster failure modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto-discovery
  python -m src.trajectory_analysis.failure_mode.extractor --traj_directory ./trajectories

  # With fixed cluster count
  python -m src.trajectory_analysis.failure_mode.extractor --traj_directory ./trajectories --k 5

  # Specify timestamps
  python -m src.trajectory_analysis.failure_mode.extractor --traj_directory ./trajectories --timestamps 2024-01-01 2024-01-02
        """,
    )
    parser.add_argument(
        "--traj_directory",
        type=str,
        default="./src/trajectory_analysis/failure_mode/sample_trajectories",
        help="Path to the root directory containing trajectory files (default: sample_trajectories)",
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default="./results/summary",
        help="Directory to write the clustered CSV outputs (default: results/summary)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model for title embeddings (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Optional fixed number of clusters (if omitted, silhouette chooses K)",
    )
    parser.add_argument(
        "--timestamps",
        nargs="*",
        default=None,
        help="Optional list of timestamps to process. If omitted, auto-discovers all subfolders.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature parameter (default: 0.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["claude", "llama", "granite"],
        default="claude",
        help="LLM model to use (default: claude)",
    )

    args = parser.parse_args()

    # Configure LLM backend
    llm_backend = None  # None = use default Claude 4 Sonnet
    if args.model == "llama":
        from src.llm.litellm import LiteLLMBackend

        llm_backend = LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct")
        print("Using Llama 3.3 70B")
    elif args.model == "granite":
        from src.llm.litellm import LiteLLMBackend

        llm_backend = LiteLLMBackend("watsonx/ibm/granite-13b-instruct-v2")
        print("Using Granite")
    else:
        print("Using Claude 4 Sonnet (default)")

    try:
        results = run_extraction_pipeline(
            traj_directory=args.traj_directory,
            summary_dir=args.summary_dir,
            model_name=args.model_name,
            k=args.k,
            timestamps=args.timestamps,
            llm_backend=llm_backend,
            temperature=args.temperature,
            verbose=True,
        )

        print("\n✅ Success!")
        print(f"   Trajectories processed: {len(results['generation']['combined_df'])}")
        print(f"   Clusters created: {results['reduction']['k']}")
        print(f"   Output files: {results['reduction']['paths']}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


# Made with Bob
