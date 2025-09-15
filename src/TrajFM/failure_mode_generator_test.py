from failure_mode_generator import process_trajectories

res = process_trajectories(
    traj_root_base="/Users/jzhou/work/notebooks/agenticfram/trajectories_codabench",
    model_id=18,
)

print(res["combined_path"])
print(res["combined_df"].head())
