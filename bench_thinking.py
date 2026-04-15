"""Benchmark thinking vs non-thinking mode on the vLLM endpoint."""

import os
import time
import json
import httpx

BASE_URL = os.environ.get("LITELLM_BASE_URL", "https://m398a4q87ru0z8-8000.proxy.runpod.net/v1").rstrip("/")
API_KEY = os.environ.get("LITELLM_API_KEY", "sk-runpod-dummy")
MODEL = "google/gemma-4-31B-it"

QUESTIONS = [
    "A centrifugal chiller at a commercial building has been showing a gradual COP degradation over the past 3 months. The evaporator approach temperature has increased by 2°C and condenser approach temperature by 1.5°C. Refrigerant charge is nominal. What are the most likely root causes and what diagnostic steps would you recommend?",
    "An industrial pump is exhibiting intermittent cavitation despite operating well above the NPSH required. Suction pressure readings are normal, but vibration spectral analysis shows sub-synchronous frequency components at 0.3x running speed. What failure mechanisms could explain this pattern and how would you differentiate between them?",
    "A variable speed drive on an AHU supply fan has been tripping on overcurrent faults only during morning startup between 6–8 AM in winter months. Motor insulation resistance tests pass. What systematic diagnostic approach would you use to isolate whether this is a thermal, electrical, or mechanical issue?",
    "A bearing on a 500 kW induction motor shows elevated temperature (85°C vs normal 65°C) and increased broadband noise floor in vibration spectrum, but no clear defect frequencies. Lubrication was performed 2 weeks ago. The motor drives a compressor with belt transmission. What are the possible causes and how would you prioritize investigation?",
    "Two identical cooling towers at the same facility show significantly different energy consumption — Tower A uses 15% more fan power than Tower B for the same heat rejection duty. Both have been recently cleaned and have similar fill condition. What variables would you investigate and what measurements would help you quantify the efficiency gap?",
]

SYSTEM_PROMPT = "You are an expert in industrial asset operations and maintenance. Provide concise but thorough analysis."


def call(question: str, thinking: bool) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": 1024,
    }
    if thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    start = time.perf_counter()
    resp = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "thinking": thinking,
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning": choice["message"].get("reasoning"),
        "content_preview": choice["message"]["content"][:120],
    }


def main():
    if not BASE_URL:
        raise SystemExit("Set LITELLM_BASE_URL environment variable.")

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"Q{i}: {q[:80]}...")
        for thinking in (False, True):
            label = "thinking" if thinking else "standard"
            print(f"  [{label}] running...", end="", flush=True)
            r = call(q, thinking)
            r["question"] = i
            results.append(r)
            has_reasoning = bool(r["reasoning"])
            print(f" {r['elapsed_s']:.3f}s | {r['completion_tokens']} tokens | reasoning={'yes' if has_reasoning else 'no'}")

    print(f"\n{'='*60}")
    print(f"{'Q':<4} {'standard (s)':<14} {'thinking (s)':<14} {'delta (s)':<12} {'reasoning?'}")
    print("-" * 60)
    for i in range(len(QUESTIONS)):
        std = results[i * 2]
        thi = results[i * 2 + 1]
        delta = thi["elapsed_s"] - std["elapsed_s"]
        has_r = "yes" if thi["reasoning"] else "no"
        print(f"Q{i+1:<3} {std['elapsed_s']:<14.3f} {thi['elapsed_s']:<14.3f} {delta:+.3f}       {has_r}")

    with open("bench_thinking_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to bench_thinking_results.json")


if __name__ == "__main__":
    main()
