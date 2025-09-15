from failure_mode_generator import process_trajectories  # Step 1 (generation)
from failure_mode_reduction import failure_mode_reduction  # Step 2 (reduction)


def run_failure_mode_pipeline(
    traj_root_base: str,
    model_id: int = 18,
    timestamps=None,  # None => auto-discover subfolders
    summary_dir: str = "summary",
    model_name: str = "all-MiniLM-L6-v2",
    k: int | None = None,  # fix cluster count if you want
):
    # Step 1: generate + save combined pickle
    gen = process_trajectories(
        timestamps=timestamps,  # or leave None to auto-discover
        traj_root_base=traj_root_base,
        model_id=model_id,
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

    # Return both results if you want to assert on them in tests
    return {"generation": gen, "reduction": red}
