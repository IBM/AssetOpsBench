#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "python-dotenv",
# ]
# ///
"""
Command-line script to run failure mode analysis on agent trajectories.

This script uses `uv` for dependency management and execution.
Supports both simple detection (Phase 2) and complete analysis with clustering (Phase 3).

Usage:
    # Simple detection only (default)
    uv run src/trajectory_analysis/failure_mode/analyze_trajectories.py

    # With clustering enabled
    uv run src/trajectory_analysis/failure_mode/analyze_trajectories.py --cluster

    # Complete example with all options
    uv run src/trajectory_analysis/failure_mode/analyze_trajectories.py \
        --path ./data \
        --cluster \
        --num-clusters 5 \
        --temperature 0.7 \
        --model claude

Or make it executable and run directly:
    chmod +x src/trajectory_analysis/failure_mode/analyze_trajectories.py
    ./src/trajectory_analysis/failure_mode/analyze_trajectories.py --cluster

Traditional Python usage (if uv not available):
    python src/trajectory_analysis/failure_mode/analyze_trajectories.py --cluster
"""

import sys
import argparse
from pathlib import Path

# Support both direct script execution and module execution
try:
    from src.trajectory_analysis.failure_mode.core import run_failure_mode_pipeline
except ModuleNotFoundError:
    # When run as a script, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.trajectory_analysis.failure_mode.core import run_failure_mode_pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run failure mode analysis on agent trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default path
  python %(prog)s
  
  # Specify custom path
  python %(prog)s --path /path/to/trajectories
  
  # With custom temperature
  python %(prog)s --path ./data --temperature 0.7
  
  # Specify output directory
  python %(prog)s --path ./data --output ./my_results
        """,
    )

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="./src/trajectory_analysis/failure_mode/sample_trajectories",
        help="Path to directory containing trajectory JSON files (default: ./src/trajectory_analysis/failure_mode/sample_trajectories)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./src/trajectory_analysis/failure_mode/processed_trajectories",
        help="Output directory for results (default: ./src/trajectory_analysis/failure_mode/processed_trajectories)",
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (0.0=deterministic, higher=more creative) (default: 0.0)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["claude", "llama", "granite"],
        default="claude",
        help="LLM model to use (default: claude)",
    )

    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Enable clustering of additional failure modes (Phase 3)",
    )

    parser.add_argument(
        "-k",
        "--num-clusters",
        type=int,
        default=None,
        help="Fixed number of clusters (default: auto-select using silhouette score)",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for clustering (default: all-MiniLM-L6-v2)",
    )

    return parser.parse_args()


def main():
    """Run failure mode analysis on test data."""

    # Parse command line arguments
    args = parse_args()

    trajectory_dir = args.path
    output_dir = args.output
    temperature = args.temperature

    print("=" * 60)
    print("Failure Mode Analysis - Example Script")
    print("=" * 60)

    # Check if trajectory directory exists
    if not Path(trajectory_dir).exists():
        print(f"\n❌ Error: Directory '{trajectory_dir}' not found!")
        print(f"   Please create it and add trajectory JSON files.")
        print(f"\n   Example trajectory format:")
        print(
            """   {
     "text": "Your question here",
     "trajectory": [
       {
         "task_description": "Step description",
         "agent_name": "AgentName",
         "response": "Agent response",
         "final_answer": "Final answer"
       }
     ]
   }"""
        )
        sys.exit(1)

    # Check if directory has files (JSON validation happens in _load_all_json_files)
    all_files = list(Path(trajectory_dir).rglob("*"))
    file_count = len([f for f in all_files if f.is_file()])

    if file_count == 0:
        print(f"\n❌ Error: No files found in '{trajectory_dir}'")
        sys.exit(1)

    print(f"\n📁 Trajectory directory: {trajectory_dir}")
    print(f"📄 Found {file_count} files (will attempt to parse as JSON)")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📂 Output directory: {output_dir}")

    print("\n🚀 Starting failure mode analysis...")
    print("   (This will use Claude 4 Sonnet by default)")
    print("   (Make sure .env file has LITELLM_API_KEY and LITELLM_BASE_URL)")

    try:
        # Configure LLM backend based on model choice
        llm_backend = None  # None = use default Claude 4 Sonnet

        if args.model == "llama":
            from src.llm.litellm import LiteLLMBackend

            llm_backend = LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct")
            print("   Using Llama 3.3 70B")
        elif args.model == "granite":
            from src.llm.litellm import LiteLLMBackend

            llm_backend = LiteLLMBackend("watsonx/ibm/granite-13b-instruct-v2")
            print("   Using Granite")
        else:
            print("   Using Claude 4 Sonnet (default)")

        # Run the pipeline (with or without clustering)
        if args.cluster:
            print("\n🔄 Clustering enabled - running complete pipeline...")
            results = run_failure_mode_pipeline(
                traj_root_base=trajectory_dir,
                llm_backend=llm_backend,
                temperature=temperature,
                summary_dir=f"{output_dir}/summary",
                model_name=args.embedding_model,
                k=args.num_clusters,
            )
            # Get results
            df = results["generation"]["combined_df"]
            combined_path = results["generation"]["combined_path"]
            has_clustering = True
        else:
            print("\n🔍 Detection only - skipping clustering...")
            from src.trajectory_analysis.failure_mode.core import process_trajectories

            gen_results = process_trajectories(
                traj_root_base=trajectory_dir,
                llm_backend=llm_backend,
                temperature=temperature,
                out_dir=output_dir,
            )
            df = gen_results["combined_df"]
            combined_path = gen_results["combined_path"]
            results = {"generation": gen_results}
            has_clustering = False

        print("\n" + "=" * 60)
        print("✅ Analysis Complete!")
        print("=" * 60)
        print(f"\n📊 Processed {len(df)} trajectories")
        print(f"💾 Results saved to:")
        print(f"   - Pickle: {combined_path}")
        print(f"   - CSV: {combined_path.replace('.pkl', '.csv')}")

        # Show failure mode summary
        print("\n📋 Failure Mode Summary:")
        print("-" * 60)

        failure_mode_columns = [
            col for col in df.columns if "." in col and col[0].isdigit()
        ]

        if failure_mode_columns:
            for mode in failure_mode_columns:
                count = df[mode].sum()
                percentage = (count / len(df)) * 100 if len(df) > 0 else 0
                status = "⚠️ " if count > 0 else "✅"
                print(f"{status} {mode}: {count}/{len(df)} ({percentage:.1f}%)")
        else:
            print("   No failure modes detected")

        # Show additional failure modes
        total_additional = df["addi_fm_cnt"].sum()
        if total_additional > 0:
            print(f"\n🔍 Additional failure modes found: {total_additional}")

            # Show examples
            with_additional = df[df["addi_fm_cnt"] > 0]
            if len(with_additional) > 0:
                print("\n   Examples:")
                for idx, row in with_additional.head(3).iterrows():
                    print(
                        f"   - Trajectory {row['ut_id']}: {row['addi_fm_cnt']} additional modes"
                    )

        # Show clustering results if enabled
        if has_clustering and "reduction" in results:
            red = results["reduction"]
            print(f"\n📊 Clustering Results:")
            print(f"   Created {red['k']} clusters")
            print(f"   Output files:")
            for key, path in red["paths"].items():
                if path:
                    print(f"   - {key}: {path}")

        print("\n" + "=" * 60)
        print("💡 Next Steps:")
        print("=" * 60)
        print(f"1. Load results: df = pd.read_pickle('{combined_path}')")
        print("2. Analyze: df.describe(), df.head()")
        print("3. Filter: df[df['1.1 Disobey Task Specification'] == True]")
        if has_clustering:
            print(
                "4. View clusters: pd.read_csv('summary/additional_fm_clustered.csv')"
            )
        else:
            print("4. Enable clustering: add --cluster flag")
        print("\n")

        return 0

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("\nCommon issues:")
        print("1. Missing .env file with API keys")
        print("2. Invalid JSON format in trajectory files")
        print("3. Network/API connection issues")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
