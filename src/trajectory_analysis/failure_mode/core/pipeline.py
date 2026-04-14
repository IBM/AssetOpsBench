"""High-level pipeline for failure mode analysis.

This module provides the main pipeline function that orchestrates trajectory
processing and failure mode detection using LLM backends.
"""

from typing import Optional

from src.llm.base import LLMBackend
from src.llm.litellm import LiteLLMBackend
from .generator import process_trajectories, combine_all_runs
from .reducer import failure_mode_reduction


def run_failure_mode_pipeline(
    traj_root_base: str,
    llm_backend: Optional[LLMBackend] = None,
    temperature: float = 0.0,
    enable_clustering: bool = False,
    runs_dir: str = "./src/trajectory_analysis/failure_mode/results/runs",
    summary_dir: str = "./src/trajectory_analysis/failure_mode/results/summary",
    model_name: str = "all-MiniLM-L6-v2",
    k: int | None = None,
    model_id: Optional[str] = None,
):
    """
    Run the failure mode analysis pipeline.

    Workflow:
    1. Analyze trajectories → saves to results/runs/{timestamp}/failure_modes.{pkl,csv}
    2. If clustering enabled:
       a. Combine all runs → results/summary/combined_failure_modes.{pkl,csv}
       b. Cluster additional failures → results/summary/additional_fm*.csv

    Args:
        traj_root_base: Root directory containing trajectory files
        llm_backend: LLM backend to use (defaults to Claude 4 Sonnet if None)
        temperature: Temperature parameter for LLM generation (default: 0.0)
        enable_clustering: Enable clustering of additional failure modes (default: False)
        runs_dir: Directory for individual run results (default: results/runs)
        summary_dir: Directory for combined/clustered results (default: results/summary)
        model_name: Sentence transformer model name for clustering
        k: Fixed cluster count (None = auto-determine optimal k)

    Returns:
        Dictionary containing:
            - generation: Results from trajectory processing
            - combination: Results from combining runs (if clustering enabled)
            - reduction: Results from clustering (if clustering enabled)

    Example:
        >>> # Simple analysis (no clustering)
        >>> results = run_failure_mode_pipeline(
        ...     traj_root_base="/path/to/trajectories"
        ... )

        >>> # With clustering
        >>> results = run_failure_mode_pipeline(
        ...     traj_root_base="/path/to/trajectories",
        ...     enable_clustering=True
        ... )
    """
    # Default to Claude Sonnet if no backend provided
    if llm_backend is None:
        llm_backend = LiteLLMBackend("litellm_proxy/claude-sonnet-4-6")
        print("Using default LLM: Claude Sonnet 4.6 (litellm_proxy/claude-sonnet-4-6)")

    # Step 1: Process trajectories and save to individual run folder
    gen = process_trajectories(
        traj_root_base=traj_root_base,
        llm_backend=llm_backend,
        temperature=temperature,
        out_dir=runs_dir,
        model_id=model_id,
    )

    result = {"generation": gen}

    # Step 2 & 3: Combine and cluster (if enabled)
    if enable_clustering:
        # Combine all runs
        combined = combine_all_runs(runs_dir=runs_dir, summary_dir=summary_dir)
        result["combination"] = combined

        if combined["combined_path"]:
            # Cluster additional failure modes
            red = failure_mode_reduction(
                combined_pickle_path=combined["combined_path"],
                out_dir=summary_dir,
                model_name=model_name,
                k=k,
            )
            result["reduction"] = red
            print(f"\n✅ Clustering complete: {red['k']} clusters")
        else:
            print("\n⚠️  No runs to cluster")

    return result
