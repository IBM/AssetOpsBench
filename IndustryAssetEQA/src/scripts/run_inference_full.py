# src/scripts/run_inference_pdm.py
import os
import json
import time
import argparse
from openai import OpenAI

from src.utils.episodic_store import EpisodicStore
from src.utils.prompt_builder_static import load_qa_index, build_prompt_for_qa

# Make sure OPENAI_API_KEY is set in your environment
client = OpenAI(api_key=os.environ.get("api_key"), base_url=os.environ.get("base_url"))

"""
<<<<<<< Updated upstream
=======
Done
>>>>>>> Stashed changes
#1. python -m src.scripts.run_inference_full --start 0 --end 5716
#DB_PATH = "data/outputs/pdm/pdm_episodic_store.db"
#QA_PATH = "data/outputs/pdm/pdm_qas_diagnostic.jsonl"
#OUT_PATH = "data/outputs/pdm/preds_pdm_qas_diagnostic.jsonl"
"""

"""
Done
#2. python -m src.scripts.run_inference_full --start 0 --end 5716
DB_PATH = "data/outputs/pdm/pdm_episodic_store.db"
QA_PATH = "data/outputs/pdm/pdm_qas_temporal.jsonl"
OUT_PATH = "data/outputs/pdm/preds_pdm_qas_temporal.jsonl"
"""

#3. python -m src.scripts.run_inference_full --start 0 --end 5716
# Done
"""
DB_PATH = "data/outputs/pdm/pdm_episodic_store.db"
QA_PATH = "data/outputs/pdm/pdm_qas_descriptive.jsonl"
OUT_PATH = "data/outputs/pdm/preds_pdm_qas_descriptive.jsonl"
"""

#4. python -m src.scripts.run_inference_full --start 0 --end 761
#Done
"""
DB_PATH = "data/outputs/pdm/pdm_episodic_store.db"
QA_PATH = "data/outputs/pdm/pdm_qa_counterfactual.jsonl"
OUT_PATH = "data/outputs/pdm/preds_pdm_qa_counterfactual.jsonl"
"""

#5. python -m src.scripts.run_inference_full --start 0 --end 902
#Done
"""
DB_PATH = "data/outputs/pdm/pdm_episodic_store.db"
QA_PATH = "data/outputs/pdm/pdm_qa_action.jsonl"
OUT_PATH = "data/outputs/pdm/preds_pdm_qa_action.jsonl"
"""

"""
#6. python -m src.scripts.run_inference_full --start 0 --end 2205
DB_PATH = "data/outputs/hydraulic/episodic_store_hyd.db"
QA_PATH = "data/outputs/hydraulic/hyd_diag_descriptive.jsonl"
OUT_PATH = "data/outputs/hydraulic/preds_hyd_diag_descriptive.jsonl"
"""

"""
#7. python -m src.scripts.run_inference_full --start 0 --end 2205
DB_PATH = "data/outputs/hydraulic/episodic_store_hyd.db"
QA_PATH = "data/outputs/hydraulic/hyd_diag_diagnostic.jsonl"
OUT_PATH = "data/outputs/hydraulic/preds_hyd_diag_diagnostic.jsonl"
"""

"""
#8. python -m src.scripts.run_inference_full --start 0 --end 10
DB_PATH = "data/outputs/hydraulic/episodic_store_hyd.db"
QA_PATH = "data/outputs/hydraulic/hyd_qas_temporal_count.jsonl"
OUT_PATH = "data/outputs/hydraulic/preds_hyd_qas_temporal_count.jsonl"
"""

"""
#9. python -m src.scripts.run_inference_full --start 0 --end 2184
#Done
DB_PATH = "data/outputs/hydraulic/episodic_store_hyd.db"
QA_PATH = "data/outputs/hydraulic/hyd_cf.jsonl"
OUT_PATH = "data/outputs/hydraulic/preds_hyd_cf.jsonl"
"""

"""
#10. python -m src.scripts.run_inference_full --start 0 --end 2205
DB_PATH = "data/outputs/hydraulic/episodic_store_hyd.db"
QA_PATH = "data/outputs/hydraulic/hyd_action.jsonl"
OUT_PATH = "data/outputs/hydraulic/preds_hyd_action.jsonl"
"""


def call_llm_and_parse_json(system: str, user: str) -> dict:
    """
    Call the LLM and parse a JSON object with keys:
      direct_answer, reasoning_answer, provenance, confidence
    """
    response = client.chat.completions.create(
        model="GCP/claude-4-sonnet",
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


def read_existing_qids(out_path: str) -> set:
    """Return set of qa_ids already written in OUT_PATH (for resume support)."""
    if not os.path.exists(out_path):
        return set()
    seen = set()
    with open(out_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("qa_id")
                if qid is not None:
                    seen.add(qid)
            except Exception:
                # ignore corrupt lines
                continue
    return seen


def robust_call(system: str, user: str, max_retries: int = 3, base_backoff: float = 2.0):
    """
    Call LLM with simple exponential backoff retries for transient failures.
    Raises last exception on permanent failure.
    """
    attempt = 0
    while True:
        try:
            return call_llm_and_parse_json(system, user)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            backoff = base_backoff * (2 ** (attempt - 1))
            print(f"[WARN] LLM call failed (attempt {attempt}/{max_retries}). "
                  f"Retrying in {backoff:.1f}s. Error: {e}")
            time.sleep(backoff)


def main(start_idx: int = None, end_idx: int = None, max_items: int = None):
    store = EpisodicStore(DB_PATH)
    qa_index = load_qa_index(QA_PATH)  # expected to return dict-like {qa_id: qa_obj}
    qa_items = list(qa_index.items())

    if not qa_items:
        store.close()
        raise RuntimeError("No QA items found in QA_PATH")

    # optional slicing
    if start_idx is not None or end_idx is not None:
        start = start_idx or 0
        end = end_idx or len(qa_items)
        qa_items = qa_items[start:end]

    if max_items is not None:
        qa_items = qa_items[:max_items]

    # resume support: skip qa_ids already in output
    seen_qids = read_existing_qids(OUT_PATH)
    total = len(qa_items)
    written = 0
    failed = 0

    print(f"Running inference for {total} QA items (output -> {OUT_PATH})")
    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        for idx, (qa_id, qa) in enumerate(qa_items, start=1):
            if qa_id in seen_qids:
                print(f"[{idx}/{total}] Skipping {qa_id} (already in output).")
                continue

            print(f"[{idx}/{total}] Processing qa_id={qa_id} ...")
            try:
                prompts = build_prompt_for_qa(store, qa)
                ans = robust_call(
                    system=prompts["system"],
                    user=prompts["user"],
                    max_retries=3,
                    base_backoff=2.0,
                )

                out_obj = {"qa_id": qa_id, "answer": ans}
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())  # force disk write
                written += 1
                print(f"[OK] Wrote prediction for {qa_id}")
            except Exception as e:
                failed += 1
                # write an error record (optional) so you can inspect failures later
                err_obj = {"qa_id": qa_id, "error": str(e)}
                fout.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[ERROR] Failed for {qa_id}: {e}")

    store.close()
    print(f"Finished. wrote={written}, failed={failed}, total={total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference over full QA dataset.")
    parser.add_argument("--start", type=int, help="start index (inclusive)", default=None)
    parser.add_argument("--end", type=int, help="end index (exclusive)", default=None)
    parser.add_argument("--max", type=int, help="max number of items to process", default=None)
    args = parser.parse_args()

    main(start_idx=args.start, end_idx=args.end, max_items=args.max)
