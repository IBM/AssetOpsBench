"""Core analysis modules for failure mode detection.

This package contains the core functionality for trajectory analysis:
- generator: LLM-based failure mode generation
- pipeline: High-level orchestration
- reducer: Clustering and categorization
- extractor: Python API for programmatic use
- utils: Helper functions
- prompts: LLM system prompts
"""

from .generator import process_trajectories
from .pipeline import run_failure_mode_pipeline
from .reducer import failure_mode_reduction
from .extractor import run_extraction_pipeline

__all__ = [
    "process_trajectories",
    "run_failure_mode_pipeline",
    "failure_mode_reduction",
    "run_extraction_pipeline",
]

# Made with Bob
