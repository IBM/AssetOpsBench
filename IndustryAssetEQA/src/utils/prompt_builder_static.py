# src/utils/prompt_builder_static.py
"""
Prompt builder for episodic QA over EpisodicStore.

Given:
  - episodic_store.db
  - a QA instance (from c_qa.jsonl, pdm_qa_*.jsonl, etc.)
it builds a prompt that:
  - presents the evidence for the fact (including time window if present)
  - explains the task_type (diagnostic, descriptive, temporal_count, counterfactual,
    action_recommendation, etc.)
  - asks the question
  - instructs the LLM to answer in strict JSON with keys:
      direct_answer, reasoning_answer, provenance, confidence
    (optionally with extra keys like "counterfactual" if the model wants to
     mirror the gold QA schema).

CLI example:

python -m src.utils.prompt_builder_static \
  --db data/episodic_store.db \
  --qa data/pdm_qa_diag.jsonl \
  --qa-id pdm_diag_pdm_m56_comp3_2015-01-02T03
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.utils.episodic_store import EpisodicStore


def load_qa_index(qa_path: str) -> Dict[str, Dict[str, Any]]:
    """Load QA JSONL and index by qa_id."""
    idx: Dict[str, Dict[str, Any]] = {}
    with open(qa_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("qa_id")
            if not qid:
                continue
            idx[qid] = obj
    return idx


def format_evidence_block(fact: Dict[str, Any], max_features: int = 20) -> str:
    """
    Render a generic evidence block for a fact from EpisodicStore.
    Works for both static (USM) and time-series (PdM) facts.
    """
    asset_id = fact.get("asset_id", fact.get("machineID", "asset"))
    label = fact.get("label", "Unknown")
    fact_id = fact.get("fact_id", "unknown_fact")
    source_file = fact.get("source_file", fact.get("dataset", "unknown"))
    row_index = fact.get("row_index", "unknown")

    start_time = fact.get("start_time")
    end_time = fact.get("end_time")

    features = fact.get("features", [])[:max_features]

    lines: List[str] = []
    lines.append(f"fact_id: {fact_id}")
    lines.append(f"asset_id: {asset_id}")
    lines.append(f"source_file: {source_file}")
    lines.append(f"row_index: {row_index}")

    if start_time and end_time:
        lines.append(f"window_start: {start_time}")
        lines.append(f"window_end: {end_time}")

    lines.append(f"dataset_label: {label}")
    lines.append("")
    lines.append("diagnostic_features:")
    for feat in features:
        name = feat.get("name")
        val = feat.get("value")
        lines.append(f"  - {name}: {val}")

    return "\n".join(lines)


def task_description_from_type(task_type: str) -> str:
    """
    Map task_type to short natural language instructions.
    """
    if task_type == "descriptive":
        return (
            "You must answer descriptive questions about the episode, such as "
            "reporting the value or summary of specific features (e.g., average "
            "vibration). Use only the evidence; do not guess values that do not appear."
        )
    elif task_type == "temporal_count":
        return (
            "You must answer temporal or counting questions (e.g., how many errors, "
            "how many distinct error types) based on the features and time window "
            "shown in the evidence. Use the numeric features provided."
        )
    elif task_type == "diagnostic":
        return (
            "You must answer diagnostic questions explaining why this episode has "
            "the given label. Use specific feature names and values from the evidence "
            "to support your explanation."
        )
    elif task_type == "counterfactual":
        return (
            "You must answer counterfactual questions about how risk would change "
            "under a specified intervention (e.g., performing maintenance earlier). "
            "If the prompt or context mentions simulator outputs (risk before and after), "
            "your answer must be consistent with them. Be explicit about the direction "
            "of risk change (increase, decrease, or no change)."
        )
    elif task_type == "action_recommendation":
        return (
            "You must recommend a high-level maintenance action (e.g., open a work order "
            "now vs continue monitoring) based on the risk implied by the features in the "
            "evidence. Refer to the features that influence your decision."
        )
    else:
        # Fallback
        return (
            "You must answer questions about this episode using only the evidence. "
            "Cite specific feature names and values in your reasoning."
        )


def build_prompt_for_qa(
    store: EpisodicStore,
    qa: Dict[str, Any],
    system_role: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build a prompt (system + user) for a single QA instance.

    Returns:
      {"system": system_prompt, "user": user_prompt}
    """
    fact_id = qa["fact_id"]
    fact = store.get_fact(fact_id)
    if fact is None:
        raise ValueError(f"Fact {fact_id} not found in EpisodicStore.")

    evidence_block = format_evidence_block(fact)
    question = qa["question"]
    task_type = qa.get("task_type", "diagnostic")

    task_desc = task_description_from_type(task_type)

    system_prompt = system_role or (
        "You are an industrial asset maintenance assistant. "
        "You MUST base your reasoning only on the provided evidence. "
        "Do NOT invent features, events, or causes that are not in the evidence. "
        "Always return a single JSON object with keys: "
        "direct_answer, reasoning_answer, provenance, confidence. "
        "If the question is counterfactual or action-oriented, you may include an "
        "additional key (e.g., 'counterfactual') if it helps structure your answer, "
        "but you must still include the four required keys."
    )

    user_prompt = f"""You are given evidence for a single episode from an industrial asset
(e.g., a machine) and a question about that episode.

TASK TYPE:
{task_type}

TASK DESCRIPTION:
{task_desc}

EVIDENCE:
{evidence_block}

QUESTION:
{question}

RESPONSE FORMAT (VERY IMPORTANT):
Return ONLY a single valid JSON object with the following keys:

- "direct_answer": a short answer to the question in 1–3 sentences.
- "reasoning_answer": a more detailed explanation citing specific feature names
  and values from the evidence (and, if applicable, any simulator outputs described
  in the question or context).
- "provenance": a JSON object with:
    - "fact_id": the fact_id from the evidence,
    - "features": a list of feature names you actually used in reasoning,
    - "file": the source_file from the evidence,
    - "row": the row_index from the evidence.
- "confidence": a number between 0 and 1 indicating your confidence.

You may optionally include additional keys (for example, "counterfactual"
with fields like "direction" and "intervention") if they are directly relevant
to the question, but the four required keys must always be present.

Example JSON skeleton (do NOT copy the content, only the structure):
{{
  "direct_answer": "...",
  "reasoning_answer": "...",
  "provenance": {{
    "fact_id": "some_fact_id",
    "features": ["feature1", "feature2"],
    "file": "some_file.csv",
    "row": 0
  }},
  "confidence": 0.8
}}

Now produce your JSON answer.
"""

    return {"system": system_prompt, "user": user_prompt}


# CLI helper
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build LLM prompt for a QA instance.")
    parser.add_argument("--db", required=True, help="Path to episodic_store.db")
    parser.add_argument("--qa", required=True, help="Path to QA JSONL file (e.g. pdm_qa_diag.jsonl)")
    parser.add_argument("--qa-id", required=True, help="qa_id to build prompt for")
    args = parser.parse_args()

    store = EpisodicStore(db_path=args.db)
    qa_index = load_qa_index(args.qa)
    qa = qa_index.get(args.qa_id)
    if qa is None:
        raise SystemExit(f"qa_id {args.qa_id} not found in {args.qa}")

    prompts = build_prompt_for_qa(store, qa)
    print("=== SYSTEM PROMPT ===")
    print(prompts["system"])
    print("\n=== USER PROMPT ===")
    print(prompts["user"])
    store.close()
