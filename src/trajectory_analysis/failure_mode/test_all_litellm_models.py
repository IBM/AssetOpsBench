#!/usr/bin/env python3
"""
Test all LiteLLM proxy models from the available models list.
This tests all models shown in the proxy's error message.
"""

import sys
import subprocess

# All models from the proxy's available models list
# Organized by provider for clarity
LITELLM_MODELS = {
    "Claude (GCP)": [
        "litellm_proxy/GCP/claude-3-7-sonnet",
    ],
    "Claude (AWS)": [
        "litellm_proxy/aws/claude-sonnet-4-6",
        "litellm_proxy/aws/claude-opus-4-6",
        "litellm_proxy/aws/claude-3-5-haiku",
    ],
    "Gemini (GCP)": [
        "litellm_proxy/GCP/gemini-2.0-flash-lite",
        "litellm_proxy/GCP/gemini-2.5-flash-lite",
        "litellm_proxy/gcp/gemini-3.1-pro-preview",
        "litellm_proxy/gemini-2.5-pro",
        "litellm_proxy/gemini-2.5-flash",
    ],
    "GPT (Azure)": [
        "litellm_proxy/Azure/gpt-5-2025-08-07",
        "litellm_proxy/Azure/gpt-5-mini-2025-08-07",
        "litellm_proxy/Azure/gpt-5-nano-2025-08-07",
        "litellm_proxy/Azure/gpt-5-chat-2025-08-07",
        "litellm_proxy/Azure/gpt-4.1",
        "litellm_proxy/Azure/gpt-4.1-mini",
        "litellm_proxy/Azure/gpt-4.1-nano",
        "litellm_proxy/Azure/o4-mini",
        "litellm_proxy/azure/gpt-5.4",
        "litellm_proxy/azure/gpt-5.3-chat",
    ],
}


def test_model(model_id: str) -> tuple[str, bool, str]:
    """Test a single model and return (model_id, success, message)."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print("=" * 60)

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "src/trajectory_analysis/failure_mode/test_llm_model_connection.py",
                "--model-id",
                model_id,
                "--timeout",
                "30",
            ],
            capture_output=True,
            text=True,
            timeout=35,  # Slightly longer than the script's timeout
        )

        if result.returncode == 0:
            return (model_id, True, "✅ SUCCESS")
        else:
            # Extract error message
            error_lines = result.stdout.split("\n")
            for i, line in enumerate(error_lines):
                if "Error type:" in line:
                    error_type = line.split("Error type:")[-1].strip()
                    if (
                        i + 1 < len(error_lines)
                        and "Error message:" in error_lines[i + 1]
                    ):
                        error_msg = (
                            error_lines[i + 1].split("Error message:")[-1].strip()
                        )
                        # Truncate long error messages
                        if len(error_msg) > 80:
                            error_msg = error_msg[:80] + "..."
                        return (model_id, False, f"❌ {error_type}: {error_msg}")
            return (model_id, False, "❌ FAILED (unknown error)")

    except subprocess.TimeoutExpired:
        return (model_id, False, "⏱️ TIMEOUT (>30s)")
    except Exception as e:
        return (model_id, False, f"❌ ERROR: {str(e)}")


def main():
    """Test all LiteLLM models and print summary."""
    print("\n" + "=" * 60)
    print("Testing All Available LiteLLM Proxy Models")
    print("=" * 60)

    total_models = sum(len(models) for models in LITELLM_MODELS.values())
    print(f"\nFound {total_models} models across {len(LITELLM_MODELS)} providers\n")

    all_results = {}
    working_models = []
    failed_models = []

    for provider, models in LITELLM_MODELS.items():
        print(f"\n{'='*60}")
        print(f"Testing {provider} ({len(models)} models)")
        print("=" * 60)

        provider_results = []
        for model_id in models:
            result = test_model(model_id)
            provider_results.append(result)

            if result[1]:  # success
                working_models.append(result)
            else:
                failed_models.append(result)

        all_results[provider] = provider_results

    # Print summary by provider
    print("\n" + "=" * 60)
    print("SUMMARY BY PROVIDER")
    print("=" * 60)

    for provider, results in all_results.items():
        working = sum(1 for _, success, _ in results if success)
        total = len(results)
        print(f"\n{provider}: {working}/{total} working")
        for model_id, success, message in results:
            status = "✅" if success else "❌"
            model_name = model_id.split("/")[-1]
            print(f"  {status} {model_name}")

    # Print overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"✅ Working: {len(working_models)}/{total_models}")
    print(f"❌ Failed:  {len(failed_models)}/{total_models}")
    print("=" * 60)

    if working_models:
        print("\n🎉 Working Models:")
        for model_id, _, _ in working_models:
            print(f"  • {model_id}")
        print("\n💡 Recommended for trajectory analysis:")
        print(f'   LiteLLMBackend("{working_models[0][0]}")')
    else:
        print("\n⚠️  No LiteLLM proxy models are working!")
        print("   Recommendation: Use watsonx/meta-llama/llama-3-3-70b-instruct")

    return 0 if working_models else 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
