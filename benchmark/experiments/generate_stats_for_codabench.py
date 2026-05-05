import os
import json
import csv


def extract_trajectory_stats(root_folder, output_csv="trajectory_stats.csv"):
    rows = []

    for folder_name in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder_name)

        if not os.path.isdir(folder_path):
            continue

        trajectory_dir = os.path.join(folder_path, "trajectory")
        if not os.path.isdir(trajectory_dir):
            print(f"[SKIP] No trajectory folder: {folder_name}")
            continue

        # Find all JSON files sorted by creation time
        json_files = sorted(
            [f for f in os.listdir(trajectory_dir) if f.endswith(".json")],
            key=lambda f: os.stat(os.path.join(trajectory_dir, f)).st_birthtime
        )

        if not json_files:
            print(f"[SKIP] No JSON files in trajectory folder: {folder_name}")
            continue

        # Get creation time of Q_1_trajectory.json from output folder as baseline
        output_dir = os.path.join(folder_path, "output")
        baseline_path = os.path.join(output_dir, "Q_1_trajectory.json")

        if os.path.isfile(baseline_path):
            baseline_time = os.stat(baseline_path).st_birthtime
        else:
            print(f"[WARN] No Q_1_trajectory.json in output folder: {folder_name}, first file duration will be None")
            baseline_time = None

        # Compute creation times for all trajectory files
        file_birthtimes = [
            os.stat(os.path.join(trajectory_dir, f)).st_birthtime
            for f in json_files
        ]

        # Duration for file[0] = birthtime[0] - baseline_time
        # Duration for file[i] = birthtime[i] - birthtime[i-1]
        file_durations = {}
        for i, f in enumerate(json_files):
            if i == 0:
                if baseline_time is not None:
                    delta = file_birthtimes[0] - baseline_time
                    file_durations[f] = round(delta, 2)
                else:
                    file_durations[f] = None
            else:
                delta = file_birthtimes[i] - file_birthtimes[i - 1]
                file_durations[f] = round(delta, 2)

        for json_file in json_files:
            json_path = os.path.join(trajectory_dir, json_file)

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON {json_file} in {folder_name}: {e}")
                continue

            if "id" not in data:
                print(f"[SKIP] No 'id' key in {json_file} ({folder_name})")
                continue

            if "trajectory" not in data:
                print(f"[SKIP] No 'trajectory' key in {json_file} ({folder_name})")
                continue

            if not data["trajectory"]:
                print(f"[SKIP] Empty or null 'trajectory' in {json_file} ({folder_name})")
                continue

            record_id = data["id"]

            # Aggregate stats across all trajectory entries
            total_tokens_sent     = 0
            total_tokens_received = 0
            total_api_calls       = 0
            found_any             = False

            for entry in data["trajectory"]:
                try:
                    model_stats = entry["logs"]["info"]["model_stats"]
                    total_tokens_sent     += model_stats.get("tokens_sent",     0)
                    total_tokens_received += model_stats.get("tokens_received", 0)
                    total_api_calls       += model_stats.get("api_calls",       0)
                    found_any = True
                except (KeyError, TypeError):
                    continue

            if not found_any:
                print(f"[SKIP] No valid model_stats in {json_file} ({folder_name})")
                continue

            rows.append({
                "id"              : record_id,
                "source_folder"   : folder_name,
                "source_file"     : json_file,
                "tokens_sent"     : total_tokens_sent,
                "tokens_received" : total_tokens_received,
                "api_calls"       : total_api_calls,
                "duration_seconds": file_durations.get(json_file),
            })
            print(f"[OK] {folder_name}/{json_file} → id={record_id}, duration={file_durations.get(json_file)}s")

    if not rows:
        print("No valid data found. CSV not written.")
        return

    fieldnames = ["id", "source_folder", "source_file",
                  "tokens_sent", "tokens_received", "api_calls",
                  "duration_seconds"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to: {output_csv}")


if __name__ == "__main__":
    root = "/Users/dhaval/Documents/GitHub/CodeBenchBackEnd/compute_worker"   # <-- change this
    extract_trajectory_stats(root, output_csv="trajectory_stats.csv")