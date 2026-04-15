#!/usr/bin/env python3
"""
Diagnostic script to verify trajectory format detection and parsing.

Usage:
    cd /path/to/AssetOpsBench
    uv run python src/trajectory_analysis/failure_mode/diagnostics/verify_trajectory_import.py [trajectory_file]
    uv run python src/trajectory_analysis/failure_mode/diagnostics/verify_trajectory_import.py --show-prompt [trajectory_file]

Examples:
    # Use default trajectory file
    cd /path/to/AssetOpsBench
    uv run python src/trajectory_analysis/failure_mode/diagnostics/verify_trajectory_import.py
    
    # Verify specific Mistral trajectory
    cd /path/to/AssetOpsBench
    uv run python src/trajectory_analysis/failure_mode/diagnostics/verify_trajectory_import.py \
        src/trajectory_analysis/failure_mode/sample_trajectories/mistral-large/0001
    
    # Verify Claude trajectory
    cd /path/to/AssetOpsBench
    uv run python src/trajectory_analysis/failure_mode/diagnostics/verify_trajectory_import.py \
        src/trajectory_analysis/failure_mode/sample_trajectories/claude-4-sonnet/0402
    
    # Show full LLM prompt that would be sent
    cd /path/to/AssetOpsBench
    uv run python src/trajectory_analysis/failure_mode/diagnostics/verify_trajectory_import.py \
        src/trajectory_analysis/failure_mode/sample_trajectories/mistral-large/0011 \
        --show-prompt
    
    # Verify your own trajectory file
    cd /path/to/AssetOpsBench
    uv run python src/trajectory_analysis/failure_mode/diagnostics/verify_trajectory_import.py \
        /path/to/your/trajectory.json
"""

import json
import argparse
from pathlib import Path

from trajectory_analysis.failure_mode.core.format_handlers import (
    get_default_registry,
)
from trajectory_analysis.failure_mode.core.prompts import system_prompt


def test_trajectory_import(trajectory_path: str, show_llm_prompt: bool = False):
    """Test if a trajectory file is imported and formatted correctly."""
    print(f"\n{'='*70}")
    print(f"Testing: {trajectory_path}")
    print("=" * 70)

    # 1. Load the JSON file
    try:
        with open(trajectory_path, "r") as f:
            data = json.load(f)
        print("✅ Step 1: JSON file loaded successfully")
        print(f"   Keys in JSON: {list(data.keys())}")
    except Exception as e:
        print(f"❌ Step 1 FAILED: Could not load JSON - {e}")
        return False

    # 2. Detect format
    try:
        registry = get_default_registry()
        handler = registry.get_handler(data)
        if handler:
            print(f"✅ Step 2: Format detected - {handler.__class__.__name__}")
        else:
            print("❌ Step 2 FAILED: No handler could detect the format")
            return False
    except Exception as e:
        print(f"❌ Step 2 FAILED: Format detection error - {e}")
        return False

    # 3. Extract and format data
    try:
        formatted = registry.format_trajectory(data)
        print("✅ Step 3: Data extracted and formatted successfully")
        print(f"\n   Question: {formatted['question'][:100]}...")
        print(f"   Number of steps: {len(formatted['steps'])}")
        print(f"   Final answer: {formatted['final_answer'][:100]}...")

        # Show first step details
        if formatted["steps"]:
            step1 = formatted["steps"][0]
            print(f"\n   First Step:")
            print(f"     Thought: {step1['thought'][:80]}...")
            print(f"     Action: {step1['action'][:80]}...")
            print(f"     Observation: {step1['observation'][:80]}...")
    except Exception as e:
        print(f"❌ Step 3 FAILED: Data extraction error - {e}")
        return False

    # 4. Show what gets passed to LLM (if requested)
    if show_llm_prompt:
        try:
            print("\n" + "=" * 70)
            print("STEP 4: SHOWING WHAT GETS PASSED TO LLM")
            print("=" * 70)

            # Build the formatted trace (same logic as in utils.py)
            question = formatted["question"]
            steps = formatted["steps"]
            final_answer = formatted["final_answer"]

            formatted_steps = [f"Question: {question}"]
            for idx, step in enumerate(steps, 1):
                step_text = (
                    f"Thought {idx}: {step['thought']}\n"
                    f"Action {idx}: {step['action']}\n"
                    f"Observation {idx}: {step['observation']}\n"
                )
                formatted_steps.append(step_text)

            formatted_steps.append(f"Answer: {final_answer}")

            # Combine into trace
            final_prompt_string = "\n" + ("-" * 40 + "\n").join(formatted_steps)

            # Show the trace portion
            print("\n📝 FORMATTED TRACE (what goes into {trace} placeholder):")
            print("-" * 70)
            print(final_prompt_string[:1000])  # Show first 1000 chars
            if len(final_prompt_string) > 1000:
                print(
                    f"\n... (truncated, total length: {len(final_prompt_string)} chars)"
                )
            print("-" * 70)

            # Show full prompt structure
            full_prompt = system_prompt.format(trace=final_prompt_string)
            print("\n📋 FULL PROMPT STRUCTURE:")
            print(f"   - System prompt length: {len(system_prompt)} chars")
            print(f"   - Trace length: {len(final_prompt_string)} chars")
            print(f"   - Total prompt length: {len(full_prompt)} chars")
            print(f"\n   First 500 chars of full prompt:")
            print("-" * 70)
            print(full_prompt[:500])
            print("...")
            print("-" * 70)

        except Exception as e:
            print(f"⚠️  Could not generate LLM prompt preview: {e}")

    print("\n✅ ALL STEPS PASSED - Trajectory imported correctly!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify that a trajectory JSON file is imported and formatted correctly"
    )
    parser.add_argument(
        "trajectory_file",
        nargs="?",
        default="src/trajectory_analysis/failure_mode/sample_trajectories/mistral-large/0001",
        help="Path to trajectory JSON file (default: sample_trajectories/mistral-large/0001)",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Show what gets passed to the LLM (formatted prompt)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TRAJECTORY FORMAT DETECTION TEST")
    print("=" * 70)

    success = test_trajectory_import(
        args.trajectory_file, show_llm_prompt=args.show_prompt
    )

    if success:
        print("\n" + "=" * 70)
        print("✅ CONCLUSION: Trajectory JSON is imported properly!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ CONCLUSION: There's an issue with trajectory import")
        print("=" * 70)
