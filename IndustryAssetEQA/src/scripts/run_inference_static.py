# scripts/run_inference_static.py

import json
from src.utils.episodic_store import EpisodicStore
from src.utils.prompt_builder_static import load_qa_index, build_prompt_for_qa
import os
import json
from openai import OpenAI

# Create a single global client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

#1. python -m src.scripts.run_inference_full --start 0 --end 5716
DB_PATH = "data/outputs/pdm/pdm_episodic_store.db"
QA_PATH = "data/outputs/pdm/pdm_qas_diagnostic.jsonl"
OUT_PATH = "data/outputs/pdm/preds_pdm_qas_diagnostic.jsonl"

#2. python -m src.scripts.run_inference_full --start 0 --end 5716
DB_PATH = "data/outputs/pdm/pdm_episodic_store.db"
QA_PATH = "data/outputs/pdm/pdm_qas_temporal.jsonl"
OUT_PATH = "data/outputs/pdm/preds_pdm_qas_temporal.jsonl"

#3. python -m src.scripts.run_inference_full --start 0 --end 5716
DB_PATH = "data/outputs/pdm/pdm_episodic_store.db"
QA_PATH = "data/outputs/pdm/pdm_qas_descriptive.jsonl"
OUT_PATH = "data/outputs/pdm/preds_pdm_qas_descriptive.jsonl"

#4. python -m src.scripts.run_inference_full --start 0 --end 761
DB_PATH = "data/outputs/pdm/pdm_episodic_store.db"
QA_PATH = "data/outputs/pdm/pdm_qa_counterfactual.jsonl"
OUT_PATH = "data/outputs/pdm/preds_pdm_qa_counterfactual.jsonl"

#5. python -m src.scripts.run_inference_full --start 0 --end 902
DB_PATH = "data/outputs/pdm/pdm_episodic_store.db"
QA_PATH = "data/outputs/pdm/pdm_qa_action.jsonl"
OUT_PATH = "data/outputs/pdm/preds_pdm_qa_action.jsonl"

#6. python -m src.scripts.run_inference_full --start 0 --end 2205
DB_PATH = "data/outputs/hydraulic/hyd_risk_model.joblib"
QA_PATH = "data/outputs/hydraulic/hyd_diag_descriptive.jsonl"
OUT_PATH = "data/outputs/hydraulic/preds_hyd_diag_descriptive.jsonl"

#7. python -m src.scripts.run_inference_full --start 0 --end 2205
DB_PATH = "data/outputs/hydraulic/hyd_risk_model.joblib"
QA_PATH = "data/outputs/hydraulic/hyd_diag_diagnostic.jsonl"
OUT_PATH = "data/outputs/hydraulic/preds_hyd_diag_diagnostic.jsonl"

#8. python -m src.scripts.run_inference_full --start 0 --end 10
DB_PATH = "data/outputs/hydraulic/hyd_risk_model.joblib"
QA_PATH = "data/outputs/hydraulic/hyd_qas_temporal_count.jsonl"
OUT_PATH = "data/outputs/hydraulic/preds_hyd_qas_temporal_count.jsonl"

#9. python -m src.scripts.run_inference_full --start 0 --end 2184
DB_PATH = "data/outputs/hydraulic/hyd_risk_model.joblib"
QA_PATH = "data/outputs/hydraulic/hyd_cf.jsonl"
OUT_PATH = "data/outputs/hydraulic/preds_hyd_cf.jsonl"

#10. python -m src.scripts.run_inference_full --start 0 --end 2205
DB_PATH = "data/outputs/hydraulic/hyd_risk_model.joblib"
QA_PATH = "data/outputs/hydraulic/hyd_action.jsonl"
OUT_PATH = "data/outputs/hydraulic/preds_hyd_action.jsonl"

def call_llm_and_parse_json(system: str, user: str) -> dict:
    """
    Call the LLM and parse a JSON object with keys:
      direct_answer, reasoning_answer, provenance, confidence
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # or another model you prefer
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    # Basic sanity check: make sure required keys exist
    required_keys = {"direct_answer", "reasoning_answer", "provenance", "confidence"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"LLM output missing keys {missing}. Got: {data}")

    return data


def main():
    store = EpisodicStore(DB_PATH)
    qa_index = load_qa_index(QA_PATH)

    # Get just ONE (qa_id, qa) pair
    try:
        qa_id, qa = next(iter(qa_index.items()))
    except StopIteration:
        print("No QA items found in the index.")
        store.close()
        return

    prompts = build_prompt_for_qa(store, qa)
    model_json = call_llm_and_parse_json(
        system=prompts["system"],
        user=prompts["user"],
    )

    # Write ONLY this one result
    with open(OUT_PATH, "w") as fout:
        fout.write(json.dumps({"qa_id": qa_id, "answer": model_json}) + "\n")

    store.close()
    print(f"Wrote predictions for a single QA item to {OUT_PATH}")


if __name__ == "__main__":
    main()
