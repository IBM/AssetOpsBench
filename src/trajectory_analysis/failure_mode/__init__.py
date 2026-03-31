"""Failure mode analysis for agent trajectories.

This module provides tools for:
1. Analyzing agent trajectories using LLM-based analysis
2. Identifying predefined failure modes (14 categories)
3. Discovering additional failure modes through clustering
4. Visualizing failure mode distributions

Main components:
- pipeline: High-level API for running the complete analysis
- generator: LLM-based trajectory analysis (Phase 1)
- reducer: Clustering and categorization (Phase 2)
- visualizer: Visualization generation
- prompts: System prompts for LLM analysis
- utils: Helper functions

Example usage:
    from src.llm.litellm import LiteLLMBackend
    from src.trajectory_analysis.failure_mode import run_failure_mode_pipeline

    llm = LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct")
    results = run_failure_mode_pipeline(
        traj_root_base="/path/to/trajectories",
        llm_backend=llm,
        summary_dir="summary_codabench"
    )
"""

# Note: Actual imports will be added in Phase 2 after refactoring
# from .pipeline import run_failure_mode_pipeline
# from .generator import process_trajectories
# from .reducer import failure_mode_reduction

__all__ = [
    # "run_failure_mode_pipeline",
    # "process_trajectories",
    # "failure_mode_reduction",
]

# Made with Bob
