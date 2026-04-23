# src/scripts/run_inference_pdm.py

import os
import json
from openai import OpenAI

from src.utils.episodic_store import EpisodicStore
from src.utils.prompt_builder_static import load_qa_index, build_prompt_for_qa

# Make sure OPENAI_API_KEY is set in your environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DB_PATH = "data/pdm_episodic_store.db"
QA_PATH = "data/pdm_qa_diag_.jsonl"
OUT_PATH = "data/pdm_preds_diag_.jsonl"


def call_llm_and_parse_json(system: str, user: str) -> dict:
    """
    Call the LLM and parse a JSON object with keys:
      direct_answer, reasoning_answer, provenance, confidence
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # change if you want a different model
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    # Basic sanity check
    required_keys = {"direct_answer", "reasoning_answer", "provenance", "confidence"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"LLM output missing keys {missing}. Got: {data}")

    return data


def main():
    store = EpisodicStore(DB_PATH)
    qa_index = load_qa_index(QA_PATH)

    # ---- ONLY ONE QUESTION ----
    # Option A: just take the first QA in the file
    qa_items = list(qa_index.items())
    if not qa_items:
        raise RuntimeError("No QA items found in QA_PATH")

    qa_id, qa = qa_items[0]
    print(f"Running inference for single QA: {qa_id}")

    prompts = build_prompt_for_qa(store, qa)
    ans = call_llm_and_parse_json(
        system=prompts["system"],
        user=prompts["user"],
    )

    # Write a single-line JSONL with this prediction
    with open(OUT_PATH, "w") as fout:
        fout.write(json.dumps({"qa_id": qa_id, "answer": ans}) + "\n")

    store.close()
    print(f"Wrote 1 prediction to {OUT_PATH}")


if __name__ == "__main__":
    main()
