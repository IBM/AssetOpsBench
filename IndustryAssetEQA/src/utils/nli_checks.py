# src/utils/nli_checks.py
"""
NLI utilities with batching, caching, and optional fp16 acceleration.

Defaults:
- MODEL_NAME uses a compact/faster NLI model to improve throughput.
- BATCH_SIZE and MAX_LENGTH tuned for typical premise+hypothesis short pairs.

You can set MODEL_NAME to "facebook/roberta-large-mnli" if you need higher fidelity;
tradeoff: much slower and larger memory footprint.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Dict, Iterable

# --- Configurable params (tune these) ---
MODEL_NAME = "FacebookAI/roberta-large-mnli"  # fast default; swap if you prefer roberta-large
BATCH_SIZE = 64
MAX_LENGTH = 128
ENTAILMENT_THRESHOLD = 0.10
ALIGNMENT_THRESHOLD = 0.10

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load tokenizer & model ---
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
_model.eval()

# Use fp16 on CUDA if possible
if device.startswith("cuda"):
    try:
        _model.half()
    except Exception:
        pass

# --- Robust label mapping (id -> label lowercase) ---
if hasattr(_model.config, "id2label") and _model.config.id2label:
    ID2LABEL_INV = {int(k): v.lower() for k, v in _model.config.id2label.items()}
else:
    ID2LABEL_INV = {0: "contradiction", 1: "neutral", 2: "entailment"}

# --- Small in-memory cache for (premise, hypothesis) -> probs dict ---
_ENTAILMENT_CACHE: Dict[Tuple[str, str], Dict[str, float]] = {}

# --- Utilities ---
def _softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)

def _batch_encode(pairs: Iterable[Tuple[str, str]], max_length: int = MAX_LENGTH):
    premises, hypos = zip(*pairs) if pairs else ([], [])
    enc = _tokenizer(
        list(premises),
        list(hypos),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc.to(device)

def entailment_probs_batch(pairs: List[Tuple[str, str]], batch_size: int = BATCH_SIZE) -> List[Dict[str, float]]:
    """
    Return list of {"entailment": p_e, "neutral": p_n, "contradiction": p_c} for each pair,
    preserving input order. Uses cache to avoid repeated work.
    """
    if not pairs:
        return []

    results: List[Dict[str, float]] = [None] * len(pairs)
    uncached_pairs: List[Tuple[str,str]] = []
    uncached_indices: List[int] = []

    # Fill results from cache where possible
    for i, pair in enumerate(pairs):
        if pair in _ENTAILMENT_CACHE:
            results[i] = _ENTAILMENT_CACHE[pair]
        else:
            uncached_pairs.append(pair)
            uncached_indices.append(i)

    # Compute uncached in batches
    with torch.no_grad():
        for start in range(0, len(uncached_pairs), batch_size):
            batch_pairs = uncached_pairs[start:start + batch_size]
            enc = _batch_encode(batch_pairs)
            # autocast for fp16 on CUDA
            if device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    outputs = _model(**enc)
            else:
                outputs = _model(**enc)
            logits = outputs.logits  # (B, num_labels)
            probs = _softmax_logits(logits).cpu()  # shape (B, num_labels)
            for j in range(probs.shape[0]):
                p = probs[j]
                # Map indices -> label names using ID2LABEL_INV
                row = { ID2LABEL_INV.get(idx, f"label_{idx}"): float(p[idx]) for idx in range(p.shape[0]) }
                idx_in_results = uncached_indices[start + j]
                results[idx_in_results] = row
                # populate cache
                _ENTAILMENT_CACHE[batch_pairs[j]] = row

    # All results filled (cached or computed)
    return results

# --- High-level metric calculations ---
def entailment_pass_rate(evidence_text: str, reasoning_sentences: List[str], threshold: float = ENTAILMENT_THRESHOLD) -> float:
    if not reasoning_sentences:
        return 0.0
    pairs = [(evidence_text, s) for s in reasoning_sentences]
    probs = entailment_probs_batch(pairs)
    entailed = sum(1 for p in probs if p.get("entailment", 0.0) >= threshold)
    return entailed / len(reasoning_sentences)

def extract_conclusion_sentence(reasoning_text: str) -> str:
    import re
    sents = re.split(r'(?<=[\.\?\!])\s+', (reasoning_text or "").strip())
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return ""
    for s in reversed(sents):
        if any(cue in s.lower() for cue in ("therefore", "thus", "hence", "consequently", "so,")):
            return s
    return sents[-1]

def reasoning_answer_alignment(reasoning_text: str, direct_answer_text: str, threshold: float = ALIGNMENT_THRESHOLD):
    conclusion = extract_conclusion_sentence(reasoning_text)
    if not conclusion or not direct_answer_text:
        return False, 0.0
    probs = entailment_probs_batch([(conclusion, direct_answer_text)])
    if not probs:
        return False, 0.0
    p_ent = probs[0].get("entailment", 0.0)
    return (p_ent >= threshold), p_ent

# --- Convenience: batch helpers (for other parts of pipeline) ---
def batch_entailment_pass_rates(instances: List[dict], premise_builder, sentence_splitter, threshold: float = ENTAILMENT_THRESHOLD) -> List[float]:
    pairs = []
    mapping = []
    for inst in instances:
        premise = premise_builder(inst)
        reasoning = inst.get("reasoning_answer", "")
        sents = sentence_splitter(reasoning)
        for s in sents:
            pairs.append((premise, s))
        mapping.append(len(sents))
    probs = entailment_probs_batch(pairs)
    out = []
    idx = 0
    for count in mapping:
        if count == 0:
            out.append(0.0)
        else:
            entailed = 0
            for _ in range(count):
                if probs[idx].get("entailment", 0.0) >= threshold:
                    entailed += 1
                idx += 1
            out.append(entailed / count)
    return out

def batch_reasoning_answer_alignment(instances: List[dict], threshold: float = ALIGNMENT_THRESHOLD):
    pairs = []
    for inst in instances:
        concl = extract_conclusion_sentence(inst.get("reasoning_answer", ""))
        hyp = inst.get("direct_answer", "")
        pairs.append((concl, hyp))
    probs = entailment_probs_batch(pairs)
    out = []
    for p in probs:
        p_ent = p.get("entailment", 0.0)
        out.append((p_ent >= threshold, p_ent))
    return out

# --- Demo main (optional) ---
if __name__ == "__main__":
    evidence = "vibration_mean = 12.3; vibration_trend = 0.48; temperature = 85C"
    reasoning_sents = [
        "Vibration is elevated relative to baseline.",
        "Therefore this is likely a bearing failure."
    ]
    print("Entailment pass:", entailment_pass_rate(evidence, reasoning_sents))
