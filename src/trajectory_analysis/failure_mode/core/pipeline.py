"""High-level pipeline for failure mode analysis.

This module provides the main pipeline function that orchestrates trajectory
processing and failure mode detection using LLM backends.
"""

from typing import Optional

from src.llm.base import LLMBackend
from src.llm.litellm import LiteLLMBackend
from .generator import process_trajectories
from .reducer import failure_mode_reduction


def run_failure_mode_pipeline(
    traj_root_base: str,
    llm_backend: Optional[LLMBackend] = None,
    temperature: float = 0.0,
    timestamps=None,  # None => auto-discover subfolders
    summary_dir: str = "summary",
    model_name: str = "all-MiniLM-L6-v2",
    k: int | None = None,  # fix cluster count if you want
):
    """
    Run the complete failure mode analysis pipeline.

    Args:
        traj_root_base: Root directory containing trajectory files
        llm_backend: LLM backend to use (defaults to Claude 4 Sonnet if None)
        temperature: Temperature parameter for LLM generation (default: 0.0)
        timestamps: Optional list of timestamps to process (None = auto-discover)
        summary_dir: Output directory for summary results
        model_name: Sentence transformer model name for clustering
        k: Fixed cluster count (None = auto-determine optimal k)

    Returns:
        Dictionary containing generation and reduction results

    Example:
        >>> # Use default Claude 4 Sonnet
        >>> results = run_failure_mode_pipeline(
        ...     traj_root_base="/path/to/trajectories"
        ... )

        >>> # Use Llama 3.3 70B
        >>> from src.llm.litellm import LiteLLMBackend
        >>> llm = LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct")
        >>> results = run_failure_mode_pipeline(
        ...     traj_root_base="/path/to/trajectories",
        ...     llm_backend=llm
        ... )
    """
    # Default to AWS Claude Sonnet if no backend provided
    if llm_backend is None:
        llm_backend = LiteLLMBackend("litellm_proxy/aws/claude-sonnet-4-6")
        print(
            "Using default LLM: AWS Claude Sonnet 4.6 (litellm_proxy/aws/claude-sonnet-4-6)"
        )

    # Step 1: generate + save combined pickle
    gen = process_trajectories(
        timestamps=timestamps,  # or leave None to auto-discover
        traj_root_base=traj_root_base,
        llm_backend=llm_backend,
        temperature=temperature,
    )
    print("Combined pickle:", gen["combined_path"])
    print(gen["combined_df"].head())

    # Step 2: reduce/cluster using the combined pickle from Step 1
    red = failure_mode_reduction(
        combined_pickle_path=gen["combined_path"],
        out_dir=summary_dir,
        model_name=model_name,
        k=k,
    )
    print("Chosen K:", red["k"])
    print("Paths:", red["paths"])
    print(red["df_clustered"].head())

    return {"generation": gen, "reduction": red}
