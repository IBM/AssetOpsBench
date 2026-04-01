#!/usr/bin/env python3
"""
Test script to verify LLM model connection via LiteLLM with timeout.
Accepts model ID as command-line argument.

Usage:
    uv run python src/trajectory_analysis/failure_mode/test_llm_model_connection.py <model_id>

Examples:
    # Test Claude
    uv run python src/trajectory_analysis/failure_mode/test_llm_model_connection.py litellm_proxy/GCP/claude-3-7-sonnet

    # Test Llama
    uv run python src/trajectory_analysis/failure_mode/test_llm_model_connection.py watsonx/meta-llama/llama-3-3-70b-instruct

    # Test GPT
    uv run python src/trajectory_analysis/failure_mode/test_llm_model_connection.py litellm_proxy/Azure/gpt-4.1
"""

import sys
import argparse
import threading
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.llm.litellm import LiteLLMBackend


def test_with_timeout(func, timeout_seconds=30):
    """Run a function with a timeout."""
    result = [None]
    exception = [None]

    def wrapper():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print(f"\n⏱️  TIMEOUT: Operation took longer than {timeout_seconds} seconds")
        print("   The model API may be slow or unresponsive")
        return False

    if exception[0]:
        raise exception[0]

    return result[0]


def test_model_connection(model_id: str):
    """Test if the specified model can be called successfully."""

    print("=" * 60)
    print("Testing LLM Model Connection")
    print("=" * 60)

    print(f"\n🧪 Testing model: {model_id}")

    # Determine which credentials will be used
    if model_id.startswith("watsonx/"):
        print("   📋 Will use: WATSONX_APIKEY, WATSONX_PROJECT_ID")
    else:
        print("   📋 Will use: LITELLM_API_KEY, LITELLM_BASE_URL")

    try:
        # Initialize backend
        print("\n1. Initializing LiteLLM backend...")
        llm_backend = LiteLLMBackend(model_id)
        print("   ✅ Backend initialized")

        # Test simple prompt
        print("\n2. Sending test prompt (with 30s timeout)...")
        test_prompt = """Please respond with a simple JSON object containing:
{
  "status": "success",
  "message": "Connection working"
}"""

        response = llm_backend.generate(prompt=test_prompt, temperature=0.0)

        print(f"\n3. Response received:")
        print(f"   {response[:200]}...")  # Show first 200 chars

        # Check if response contains expected content
        if response and len(response) > 0:
            print("\n✅ SUCCESS: Model is responding correctly!")
            print(f"\n💡 Use this model in your code:")
            print(f'   LiteLLMBackend("{model_id}")')
            return True
        else:
            print("\n⚠️  WARNING: Empty response received")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: Failed to connect to model")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print("\n💡 Troubleshooting:")
        if model_id.startswith("watsonx/"):
            print("   1. Check WATSONX_APIKEY in .env file")
            print("   2. Check WATSONX_PROJECT_ID in .env file")
            print("   3. Verify WatsonX endpoint is accessible")
        else:
            print("   1. Check LITELLM_API_KEY in .env file")
            print("   2. Check LITELLM_BASE_URL in .env file")
            print("   3. Verify the LiteLLM proxy is running")
            print("   4. Confirm model is available in your proxy")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test LLM model connection via LiteLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test Claude
  python %(prog)s --model-id litellm_proxy/GCP/claude-3-7-sonnet
  
  # Test Llama
  python %(prog)s --model-id watsonx/meta-llama/llama-3-3-70b-instruct
  
  # Test GPT
  python %(prog)s --model-id litellm_proxy/Azure/gpt-4.1
  
  # Test Gemini
  python %(prog)s --model-id litellm_proxy/GCP/gemini-2.5-flash
        """,
    )

    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID with provider prefix (e.g., watsonx/..., litellm_proxy/...)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds (default: 30)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\n⏱️  Running with {args.timeout} second timeout...")
    success = test_with_timeout(
        lambda: test_model_connection(args.model_id), timeout_seconds=args.timeout
    )

    if success is False:
        print("\n❌ Test failed or timed out")
        sys.exit(1)
    elif success is True:
        print("\n✅ Test passed")
        sys.exit(0)
    else:
        print("\n⚠️  Test result unclear")
        sys.exit(1)

# Made with Bob
