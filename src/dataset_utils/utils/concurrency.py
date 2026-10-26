from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, Hashable, List, Union

from tqdm import tqdm


def concurrent_dict_execution(
    func,
    params: Dict[Hashable, Union[List[Any], Dict[str, Any]]],
    executor: str = "thread",
    num_max_workers=5,
):
    if executor == "thread":
        exe = ThreadPoolExecutor
    elif executor == "process":
        exe = ProcessPoolExecutor
    else:
        raise NotImplementedError("select executor out of thread|process")

    with exe(max_workers=num_max_workers) as _exe:
        _single_param = list(params.values())[0]
        if isinstance(_single_param, list):
            future_to_name = {_exe.submit(func, *p): k for k, p in params.items()}
        elif isinstance(_single_param, dict):
            future_to_name = {
                _exe.submit(func, **p): k  # type: ignore
                for k, p in params.items()
            }
        else:
            raise RuntimeError("Improper function use!")

        for future in tqdm(
            as_completed(future_to_name), total=len(future_to_name), desc="Completed:"
        ):
            name = future_to_name[future]

            try:
                yield name, future.result()

            except Exception as e:
                print(f"Error in {name}: {e}")
