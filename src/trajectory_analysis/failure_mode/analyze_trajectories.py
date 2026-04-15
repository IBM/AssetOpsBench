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
Supports both simple detection and complete analysis with clustering.

Usage:
    # Simple analysis with default model (LiteLLM Claude Sonnet 4.6)
    uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py

    # Specify custom trajectory path
    uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
        --path ./my_trajectories

    # With verbose logging
    uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
        --path ./my_trajectories \
        --verbose

    # Use WatsonX Llama model
    uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
        --model-id watsonx/meta-llama/llama-3-3-70b-instruct

    # Use specific LiteLLM proxy model
    uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
        --model-id litellm_proxy/claude-sonnet-4-6

    # With clustering enabled
    uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
        --cluster

    # Complete example with all options
    uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
        --path ./my_trajectories \
        --output ./my_results \
        --model-id litellm_proxy/claude-sonnet-4-6 \
        --temperature 0.0 \
        --cluster \
        --num-clusters 5 \
        --verbose

    # Cluster existing runs without new analysis
    uv run python src/trajectory_analysis/failure_mode/analyze_trajectories.py \
        --cluster-only

Traditional Python usage (if uv not available):
    python src/trajectory_analysis/failure_mode/analyze_trajectories.py --verbose
"""

import sys
import argparse
import logging
from pathlib import Path

# Import from the package
from trajectory_analysis.failure_mode.core import run_failure_mode_pipeline


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
        default="./src/trajectory_analysis/failure_mode/results",
        help="Output directory for results (default: ./src/trajectory_analysis/failure_mode/results)",
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
        "--model-id",
        type=str,
        default=None,
        help="Full model ID (e.g., 'litellm_proxy/claude-sonnet-4-6' or 'watsonx/meta-llama/llama-3-3-70b-instruct'). "
        "If not specified, uses litellm_proxy/claude-sonnet-4-6 (LiteLLM) or watsonx/meta-llama/llama-3-3-70b-instruct (WatsonX)",
    )

    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Enable clustering of additional failure modes after analysis",
    )

    parser.add_argument(
        "--cluster-only",
        action="store_true",
        help="Skip trajectory analysis, only combine existing runs and cluster (requires existing runs in --output/runs/)",
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

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows detailed file processing and handler info)",
    )

    return parser.parse_args()


def main():
    """Run failure mode analysis on test data."""

    # Parse command line arguments
    args = parse_args()

    output_dir = args.output
    temperature = args.temperature

    # Configure logging
    # Always log INFO to file, but only show on screen if --verbose
    log_level = logging.INFO

    # Console handler - only if verbose
    handlers = []
    if args.verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(console_handler)

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    print("=" * 60)
    print("Failure Mode Analysis")
    print("=" * 60)

    # Handle cluster-only mode
    if args.cluster_only:
        print("\n📊 Cluster-only mode: Combining and clustering existing runs...")
        print(f"📂 Output directory: {output_dir}")

        # Check if runs directory exists
        runs_dir = Path(output_dir) / "runs"
        if not runs_dir.exists() or not any(runs_dir.iterdir()):
            print(f"\n❌ Error: No runs found in '{runs_dir}'")
            print("   Please run trajectory analysis first without --cluster-only")
            sys.exit(1)

        # Setup logging for clustering (ALWAYS, regardless of verbose flag)
        summary_dir = Path(output_dir) / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        log_file = summary_dir / "clustering.log"

        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        print(f"📝 Logging to file: {log_file}")

        if args.verbose:
            print("🔊 Verbose mode: Showing detailed logs on screen")
        else:
            print(
                "🔇 Quiet mode: Logs saved to file only (use --verbose to see on screen)"
            )

        try:
            from trajectory_analysis.failure_mode.core.generator import (
                combine_all_runs,
            )
            from trajectory_analysis.failure_mode.core.reducer import (
                failure_mode_reduction,
            )

            # Combine all runs
            combined = combine_all_runs(
                runs_dir=str(runs_dir), summary_dir=f"{output_dir}/summary"
            )

            if not combined["combined_path"]:
                print("\n❌ Error: No valid runs to combine")
                sys.exit(1)

            # Cluster
            red = failure_mode_reduction(
                combined_pickle_path=combined["combined_path"],
                out_dir=f"{output_dir}/summary",
                model_name=args.embedding_model,
                k=args.num_clusters,
            )

            print("\n" + "=" * 60)
            print("✅ Clustering Complete!")
            print("=" * 60)
            print(f"\n📊 Combined {combined['n_runs']} runs")
            print(f"📦 Results saved to: {output_dir}/summary/")
            print(f"   - combined_failure_modes.pkl")
            print(f"   - combined_failure_modes.csv")
            print(f"   - additional_fm.csv")
            print(f"   - additional_fm_clustered.csv")
            print(f"   - {red['k']} clusters identified")
            print("\n")
            return 0

        except Exception as e:
            print(f"\n❌ Error during clustering: {e}")
            import traceback

            traceback.print_exc()
            return 1

    # Normal mode: analyze trajectories
    trajectory_dir = args.path

    # Check if trajectory directory exists
    if not Path(trajectory_dir).exists():
        print(f"\n❌ Error: Directory '{trajectory_dir}' not found!")
        print(f"   Please create it and add trajectory JSON files.")
        sys.exit(1)

    # Check if directory has files
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

    # Setup file logging (ALWAYS, regardless of verbose flag)
    file_handler = None
    run_id_for_pipeline = None

    from datetime import datetime

    # Create runs directory
    runs_dir_path = Path(output_dir) / "runs"
    runs_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate run_id (timestamp without "run_" prefix to match generator.py)
    run_id_for_pipeline = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_path = runs_dir_path / run_id_for_pipeline
    run_dir_path.mkdir(parents=True, exist_ok=True)

    # Setup file logging (always enabled for audit trail)
    log_file = run_dir_path / "analysis.log"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    print(f"📝 Logging to file: {log_file}")

    if args.verbose:
        print("🔊 Verbose mode: Showing detailed logs on screen")
    else:
        print("🔇 Quiet mode: Logs saved to file only (use --verbose to see on screen)")

    if args.verbose:
        print("🔊 Verbose mode: Showing detailed logs on screen")
    else:
        print("🔇 Quiet mode: Logs saved to file only (use --verbose to see on screen)")

    try:
        from llm.litellm import LiteLLMBackend

        # Configure LLM backend based on model-id
        if args.model_id:
            # User specified a model ID
            model_id = args.model_id
            llm_backend = LiteLLMBackend(model_id)

            # Determine which credentials are needed
            if model_id.startswith("watsonx/"):
                print(f"   Using WatsonX model: {model_id}")
                print("   (Make sure .env has WATSONX_APIKEY and WATSONX_PROJECT_ID)")
            elif model_id.startswith("litellm_proxy/"):
                print(f"   Using LiteLLM proxy model: {model_id}")
                print("   (Make sure .env has LITELLM_API_KEY and LITELLM_BASE_URL)")
            else:
                print(f"   Using model: {model_id}")
        else:
            # No model specified - use default (None = Claude 4 Sonnet in generator.py)
            llm_backend = None
            model_id = "litellm_proxy/claude-sonnet-4-6"  # Set default model_id
            print(f"   Using default: {model_id} (LiteLLM)")
            print("   (Make sure .env has LITELLM_API_KEY and LITELLM_BASE_URL)")
            print(
                "   Alternative: Use --model-id watsonx/meta-llama/llama-3-3-70b-instruct for WatsonX"
            )

        # Run the pipeline
        results = run_failure_mode_pipeline(
            traj_root_base=trajectory_dir,
            llm_backend=llm_backend,
            temperature=temperature,
            enable_clustering=args.cluster,
            runs_dir=f"{output_dir}/runs",
            summary_dir=f"{output_dir}/summary",
            model_name=args.embedding_model,
            k=args.num_clusters,
            model_id=model_id,  # Use the model_id variable instead of args.model_id
            run_id=run_id_for_pipeline,  # Pass pre-generated run_id for logging
        )

        # Get results
        gen_results = results["generation"]
        df = gen_results["df"]
        run_dir = gen_results["run_dir"]
        run_id = gen_results["run_id"]

        print("\n" + "=" * 60)
        print("✅ Analysis Complete!")
        print("=" * 60)
        print(f"\n📊 Processed {len(df)} trajectories")
        print(f"💾 Results saved to: {run_dir}")
        print(f"   - failure_modes.pkl")
        print(f"   - failure_modes.csv")

        if args.cluster and "combination" in results:
            combined_path = results["combination"]["combined_path"]
            if combined_path:
                print(f"\n📦 Combined results: {output_dir}/summary/")
                print(f"   - combined_failure_modes.pkl")
                print(f"   - combined_failure_modes.csv")

            if "reduction" in results:
                print(f"   - additional_fm.csv")
                print(f"   - additional_fm_clustered.csv")
                print(f"   - {results['reduction']['k']} clusters identified")

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

        print("\n" + "=" * 60)
        print("💡 Next Steps:")
        print("=" * 60)
        print(f"1. Load results: df = pd.read_pickle('{run_dir}/failure_modes.pkl')")
        print("2. Analyze: df.describe(), df.head()")
        print("3. Filter: df[df['1.1 Disobey Task Specification'] == True]")
        if args.cluster:
            print(
                f"4. View combined: pd.read_pickle('{output_dir}/summary/combined_failure_modes.pkl')"
            )
            print(
                f"5. View clusters: pd.read_csv('{output_dir}/summary/additional_fm_clustered.csv')"
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
    finally:
        # Close file handler if it was created
        if file_handler:
            file_handler.close()
            logging.getLogger().removeHandler(file_handler)


if __name__ == "__main__":
    sys.exit(main())

