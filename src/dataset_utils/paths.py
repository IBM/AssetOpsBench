import os
from pathlib import Path
from typing import Union

DATASET_DIR_ENV = "ASSETOPSBENCH_DATASET_DIR"
PathLike = Union[str, os.PathLike[str]]


def get_dataset_dir(data_dir: PathLike | None = None) -> Path:
    """Return the configured external dataset workspace.

    An explicit path takes precedence over ``ASSETOPSBENCH_DATASET_DIR``. The
    directory is intentionally external to the installed package because it
    may contain generated datasets and private source artifacts.
    """
    configured = data_dir if data_dir is not None else os.environ.get(DATASET_DIR_ENV)
    if not configured:
        raise RuntimeError(
            f"Set {DATASET_DIR_ENV} to the directory containing extracted, "
            "datasets, and test_datasets."
        )

    return Path(configured).expanduser().resolve()
