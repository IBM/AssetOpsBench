#!/usr/bin/env python3
"""
Example script to run failure mode analysis on trajectories.

Usage:
    # Use default path (./test_data)
    python src/trajectory_analysis/failure_mode/example.py

    # Specify custom path
    python src/trajectory_analysis/failure_mode/example.py --path /path/to/trajectories

    # With temperature control
    python src/trajectory_analysis/failure_mode/example.py --path ./data --temperature 0.7

Or make it executable:
    chmod +x src/trajectory_analysis/failure_mode/example.py
    ./src/trajectory_analysis/failure_mode/example.py --path ./my_data
"""

import sys
import argparse
from pathlib import Path
from src.trajectory_analysis.failure_mode import run_failure_mode_pipeline


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
        default="./test_data",
        help="Path to directory containing trajectory JSON files (default: ./test_data)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
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

    # Check for JSON files
    json_files = list(Path(trajectory_dir).glob("*.json"))
    if not json_files:
        print(f"\n❌ Error: No JSON files found in '{trajectory_dir}'")
        sys.exit(1)

    print(f"\n📁 Trajectory directory: {trajectory_dir}")
    print(f"📄 Found {len(json_files)} JSON files")

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

        # Run the pipeline
        results = run_failure_mode_pipeline(
            traj_root_base=trajectory_dir,
            llm_backend=llm_backend,
            temperature=temperature,
        )

        # Get results
        df = results["generation"]["combined_df"]
        combined_path = results["generation"]["combined_path"]

        print("\n" + "=" * 60)
        print("✅ Analysis Complete!")
        print("=" * 60)
        print(f"\n📊 Processed {len(df)} trajectories")
        print(f"💾 Results saved to: {combined_path}")

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
        print(f"1. Load results: df = pd.read_pickle('{combined_path}')")
        print("2. Analyze: df.describe(), df.head()")
        print("3. Filter: df[df['1.1 Disobey Task Specification'] == True]")
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
