"""Deterministic scoring for structured FMEA catalogue scenarios."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from typing import Any

from ..models import Scenario, ScorerResult
from . import register
from .static_json import flatten_answer, normalize_value, parse_structured_answer

_REQUIRED_CATALOG_TOOLS = {
    "utilities__get_asset_catalog",
    "utilities__get_failure_mode_catalog",
}
_DOMAIN_PREFIXES = ("fmsr__", "iot__", "tsfm__", "vibration__", "wo__")
_DEFAULT_F1_PASS_THRESHOLD = 0.80

# Failure-location generation (asset_locations family). Matching follows SSEE
# (Lynch et al., AAAI-25): embed both sets, keep pairs at or above a cosine
# threshold, assign greedily one-to-one. Encoder and 0.8 threshold follow
# arXiv 2509.17834. The encoder is preloaded into the Harbor base image
# (base-image/models.txt) and otherwise downloads on first use.
_DEFAULT_ENCODER = "sentence-transformers/msmarco-MiniLM-L-6-v3"
_DEFAULT_MATCH_THRESHOLD = 0.8
_DEFAULT_MAX_LOCATIONS = 30


def _strict_json_object(answer: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse one JSON object and record duplicate keys at any nesting level."""
    duplicates: list[str] = []

    def object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        counts = Counter(key for key, _ in pairs)
        duplicates.extend(sorted(key for key, count in counts.items() if count > 1))
        return dict(pairs)

    try:
        parsed = json.loads(answer.strip(), object_pairs_hook=object_pairs_hook)
    except (json.JSONDecodeError, TypeError):
        return None, []
    return (parsed if isinstance(parsed, dict) else None), duplicates


def _metadata(scenario: Scenario, key: str) -> dict[str, Any]:
    metadata = scenario.evaluation_metadata or {}
    value = metadata.get(key, {})
    return value if isinstance(value, dict) else {}


def _f1_pass_threshold(meta: dict[str, Any]) -> float | None:
    """Read f1_pass_threshold, defaulting when unset; None when it is not in [0, 1]."""
    value = meta.get("f1_pass_threshold")
    if value is None:
        return _DEFAULT_F1_PASS_THRESHOLD
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # NaN fails the range test too.
    return float(value) if 0.0 <= value <= 1.0 else None


def _weights(scenario: Scenario) -> dict[str, float]:
    defaults = {
        "coverage": 0.35,
        "precision": 0.35,
        "instruction_following": 0.20,
        "grounding": 0.10,
    }
    dimensions = _metadata(scenario, "rubric").get("dimensions", [])
    if not isinstance(dimensions, list):
        return defaults

    weights = dict(defaults)
    for dimension in dimensions:
        if not isinstance(dimension, dict):
            continue
        name, weight = dimension.get("name"), dimension.get("weight")
        if name in weights and isinstance(weight, (int, float)):
            weights[name] = float(weight)

    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()} if total else defaults


def _same_scalar(left: Any, right: Any) -> bool:
    return normalize_value(left) == normalize_value(right)


def _instruction_score(
    strict_object: dict[str, Any] | None,
    duplicates: list[str],
    expected_top_keys: set[str],
) -> tuple[float, bool, bool]:
    json_only = strict_object is not None and not duplicates
    actual_top_keys = set(strict_object) if strict_object is not None else set()
    exact_top_keys = actual_top_keys == expected_top_keys
    return (float(json_only) + float(exact_top_keys)) / 2.0, json_only, exact_top_keys


def _tool_diagnostics(trajectory_text: str) -> dict[str, Any]:
    try:
        trajectory = json.loads(trajectory_text) if trajectory_text else None
    except json.JSONDecodeError:
        trajectory = None

    names: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            name = value.get("name")
            if isinstance(name, str) and "__" in name:
                names.append(name)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(trajectory)
    unique = sorted(set(names))
    used_required = sorted(_REQUIRED_CATALOG_TOOLS.intersection(unique))
    missing_required = sorted(_REQUIRED_CATALOG_TOOLS.difference(unique))
    out_of_scope = sorted(name for name in unique if name.startswith(_DOMAIN_PREFIXES))
    return {
        "catalog_tools_used": used_required,
        "missing_catalog_tools": missing_required,
        "out_of_scope_domain_tools": out_of_scope,
        "catalog_process_compliant": not missing_required and not out_of_scope,
    }


def _score_asset_modes(
    gold: dict[str, Any], model: dict[str, Any]
) -> tuple[dict[str, float], dict[str, Any], bool]:
    gold_raw, model_raw = gold.get("failure_modes"), model.get("failure_modes")
    gold_modes = gold_raw if isinstance(gold_raw, dict) else {}
    model_modes = model_raw if isinstance(model_raw, dict) else {}
    gold_keys, model_keys = set(gold_modes), set(model_modes)
    correct_keys = gold_keys & model_keys
    correct_pairs = {
        key for key in correct_keys if _same_scalar(gold_modes[key], model_modes[key])
    }

    coverage = len(correct_keys) / len(gold_keys) if gold_keys else 1.0
    precision = len(correct_keys) / len(model_keys) if model_keys else float(not gold_keys)
    grounding = len(correct_pairs) / len(model_keys) if model_keys else float(not gold_keys)
    asset_match = (
        isinstance(model.get("asset"), str)
        and isinstance(gold.get("asset"), str)
        and model["asset"].strip().casefold() == gold["asset"].strip().casefold()
    )
    dimensions = {"coverage": coverage, "precision": precision, "grounding": grounding}
    details = {
        "asset_match": asset_match,
        "gold_modes": sorted(gold_keys),
        "returned_modes": sorted(model_keys),
        "correct_modes": sorted(correct_keys),
        "missing_modes": sorted(gold_keys - model_keys),
        "extra_modes": sorted(model_keys - gold_keys),
        "description_matches": sorted(correct_pairs),
    }
    semantic_exact = asset_match and gold_keys == model_keys and len(correct_pairs) == len(gold_keys)
    return dimensions, details, semantic_exact


class _EncoderUnavailable(RuntimeError):
    """The sentence encoder could not be imported, downloaded or run."""


@lru_cache(maxsize=2)
def _load_encoder(name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def _similarity(left: list[str], right: list[str], encoder: str) -> list[list[float]]:
    """Cosine similarity matrix, rows for ``left`` and columns for ``right``."""
    if not left or not right:
        return [[0.0] * len(right) for _ in left]
    model = _load_encoder(encoder)
    a = model.encode(left, normalize_embeddings=True, convert_to_numpy=True)
    b = model.encode(right, normalize_embeddings=True, convert_to_numpy=True)
    return (a @ b.T).tolist()


def _ssee_assign(sim: list[list[float]], threshold: float) -> dict[int, int]:
    """SSEE greedy one-to-one assignment of rows to columns.

    Each row keeps its columns at or above ``threshold``, best first. Rows are
    visited in order of their best similarity, and each takes its most similar
    column that no earlier row has taken.
    """
    ranked = []
    for row, scores in enumerate(sim):
        cands = sorted(
            ((value, col) for col, value in enumerate(scores) if value >= threshold),
            reverse=True,
        )
        if cands:
            ranked.append((cands[0][0], row, cands))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    taken: set[int] = set()
    assignment: dict[int, int] = {}
    for _, row, cands in ranked:
        for _, col in cands:
            if col not in taken:
                taken.add(col)
                assignment[row] = col
                break
    return assignment


def _number(meta: dict[str, Any], key: str, default: float, *, lo: float, hi: float) -> float | None:
    value = meta.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if lo <= value <= hi else None


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _score_asset_locations(
    gold: dict[str, Any], model: dict[str, Any], meta: dict[str, Any]
) -> tuple[dict[str, float], dict[str, Any], bool]:
    """SSEE precision and recall over failure-location names, with a length cap."""
    threshold = _number(meta, "match_threshold", _DEFAULT_MATCH_THRESHOLD, lo=0.0, hi=1.0)
    cap = _number(meta, "max_locations", _DEFAULT_MAX_LOCATIONS, lo=1, hi=10_000)
    if threshold is None or cap is None or int(cap) != cap:
        raise ValueError("match_threshold must be in [0, 1] and max_locations a positive integer")
    cap = int(cap)
    encoder = str(meta.get("encoder") or _DEFAULT_ENCODER)

    gold_raw, model_raw = gold.get("failure_locations"), model.get("failure_locations")
    gold_names = list(gold_raw) if isinstance(gold_raw, dict) else []
    returned = list(model_raw) if isinstance(model_raw, dict) else []
    model_names = returned[:cap]

    try:
        sim = _similarity(gold_names, model_names, encoder)
    except Exception as exc:  # import, download (network/proxy) or inference failure
        raise _EncoderUnavailable(f"{encoder}: {exc!r}") from exc
    by_gold = _ssee_assign(sim, threshold)
    transposed = [list(col) for col in zip(*sim)] if sim and model_names else [[] for _ in model_names]
    by_model = _ssee_assign(transposed, threshold)

    recall = len(by_gold) / len(gold_names) if gold_names else 1.0
    precision = len(by_model) / len(model_names) if model_names else float(not gold_names)

    # Same metric with the prompt's seed locations removed from both sets, so
    # restating the seeds earns no credit.
    seeds = {str(name).strip().casefold() for name in meta.get("seed_locations") or []}
    seed_gold = {i for i, name in enumerate(gold_names) if name.strip().casefold() in seeds}
    seed_model = {by_gold[i] for i in seed_gold if i in by_gold}
    novel_hits = sum(1 for i, j in by_gold.items() if i not in seed_gold)
    novel_gold = len(gold_names) - len(seed_gold)
    novel_model = len(model_names) - len(seed_model)
    novel_recall = novel_hits / novel_gold if novel_gold else 1.0
    novel_precision = novel_hits / novel_model if novel_model else float(not novel_gold)

    asset_match = (
        isinstance(model.get("asset"), str)
        and isinstance(gold.get("asset"), str)
        and model["asset"].strip().casefold() == gold["asset"].strip().casefold()
    )
    # Gold locations carry no descriptions, so grounding mirrors precision.
    dimensions = {"coverage": recall, "precision": precision, "grounding": precision}
    details = {
        "asset_match": asset_match,
        "encoder": encoder,
        "match_threshold": threshold,
        "max_locations": cap,
        "returned_count": len(returned),
        "over_cap": len(returned) > cap,
        "scored_locations": model_names,
        "matches": sorted(
            (gold_names[i], model_names[j], round(sim[i][j], 4)) for i, j in by_gold.items()
        ),
        "missing_locations": sorted(gold_names[i] for i in range(len(gold_names)) if i not in by_gold),
        "extra_locations": sorted(model_names[j] for j in range(len(model_names)) if j not in by_model),
        "seed_locations": sorted(gold_names[i] for i in seed_gold),
        "novel_recall": novel_recall,
        "novel_precision": novel_precision,
        "novel_f1": _f1(novel_precision, novel_recall),
    }
    exact = asset_match and len(by_gold) == len(gold_names) == len(model_names)
    return dimensions, details, exact


def _score_analytics(
    gold: dict[str, Any], model: dict[str, Any]
) -> tuple[dict[str, float], dict[str, Any], bool]:
    gold_flat, model_flat = flatten_answer(gold), flatten_answer(model)
    gold_keys, model_keys = set(gold_flat), set(model_flat)
    shared_keys = gold_keys & model_keys
    correct_keys = {key for key in shared_keys if gold_flat[key] == model_flat[key]}
    coverage = len(correct_keys) / len(gold_keys) if gold_keys else 1.0
    precision = len(correct_keys) / len(model_keys) if model_keys else float(not gold_keys)
    grounding = len(correct_keys) / len(shared_keys) if shared_keys else 0.0
    dimensions = {"coverage": coverage, "precision": precision, "grounding": grounding}
    details = {
        "gold_facts": gold_flat,
        "returned_facts": model_flat,
        "correct_fact_keys": sorted(correct_keys),
        "missing_fact_keys": sorted(gold_keys - model_keys),
        "extra_fact_keys": sorted(model_keys - gold_keys),
        "wrong_value_keys": sorted(shared_keys - correct_keys),
    }
    return dimensions, details, gold_flat == model_flat


def fmea_score(scenario: Scenario, answer: str, trajectory_text: str) -> ScorerResult:
    """Score FMEA content by semantic F1 and retain strict/rubric diagnostics."""
    gold = parse_structured_answer(scenario.expected_answer)
    tolerant_model = parse_structured_answer(answer)
    strict_model, duplicates = _strict_json_object(answer)

    if not isinstance(gold, dict):
        return ScorerResult(
            scorer="fmea",
            passed=False,
            score=0.0,
            rationale="FMEA ground truth is not a JSON object.",
            details={"parse_error": "invalid_ground_truth"},
        )

    model = tolerant_model if isinstance(tolerant_model, dict) else {}
    meta = _metadata(scenario, "scenario_meta")
    f1_pass_threshold = _f1_pass_threshold(meta)
    if f1_pass_threshold is None:
        return ScorerResult(
            scorer="fmea",
            passed=False,
            score=0.0,
            rationale=(
                "scenario_meta.json f1_pass_threshold must be a number in [0, 1]; "
                f"got {meta.get('f1_pass_threshold')!r}."
            ),
            details={"error": "invalid_scenario_meta"},
        )
    tag = str(meta.get("tag", ""))
    is_asset_modes = tag == "asset_modes" or (
        set(gold) == {"asset", "failure_modes"}
        and isinstance(gold.get("failure_modes"), dict)
    )
    is_asset_locations = tag == "asset_locations" or (
        set(gold) == {"asset", "failure_locations"}
        and isinstance(gold.get("failure_locations"), dict)
    )
    if is_asset_locations:
        try:
            dimensions, content_details, semantic_exact = _score_asset_locations(gold, model, meta)
        except ValueError as exc:
            return ScorerResult(
                scorer="fmea",
                passed=False,
                score=0.0,
                rationale=f"scenario_meta.json: {exc}.",
                details={"error": "invalid_scenario_meta"},
            )
        except _EncoderUnavailable as exc:
            return ScorerResult(
                scorer="fmea",
                passed=False,
                score=0.0,
                rationale=f"Location encoder unavailable: {exc}.",
                details={"error": "encoder_unavailable"},
            )
        scoring_family = "asset_locations"
    elif is_asset_modes:
        dimensions, content_details, semantic_exact = _score_asset_modes(gold, model)
        scoring_family = "asset_modes"
    else:
        dimensions, content_details, semantic_exact = _score_analytics(gold, model)
        scoring_family = "catalogue_analytics"

    instruction, json_only, exact_top_keys = _instruction_score(strict_model, duplicates, set(gold))
    dimensions["instruction_following"] = instruction
    weights = _weights(scenario)
    rubric_weighted_score = sum(weights[name] * dimensions[name] for name in weights)
    precision, recall = dimensions["precision"], dimensions["coverage"]
    semantic_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Compare at the reported precision: 2PR/(P+R) lands just under an exact
    # boundary (6/7 and 6/8 give 0.7999999999999999), failing a 0.8 threshold.
    score = round(semantic_f1, 6)
    passed = score >= f1_pass_threshold
    strict_success = semantic_exact and json_only and exact_top_keys
    baseline_f1 = _metadata(scenario, "reference_answer").get("f1")
    details: dict[str, Any] = {
        "family": scoring_family,
        "scenario_tag": tag or None,
        "dimensions": dimensions,
        "weights": weights,
        "semantic_f1": semantic_f1,
        "f1_pass_threshold": f1_pass_threshold,
        "f1_threshold_operator": ">=",
        "rubric_weighted_score": rubric_weighted_score,
        "raw_weighted_score": rubric_weighted_score,
        "strict_success": strict_success,
        "semantic_exact": semantic_exact,
        "json_only": json_only,
        "exact_top_level_keys": exact_top_keys,
        "duplicate_keys": duplicates,
        "baseline_f1_reported": baseline_f1,
        "baseline_normalization_applied": False,
        **content_details,
        **_tool_diagnostics(trajectory_text),
    }
    rationale = (
        f"Semantic F1={semantic_f1:.3f} "
        f"({'>=' if passed else '<'} {f1_pass_threshold:.2f}); "
        f"coverage={recall:.3f}, precision={precision:.3f}, "
        f"rubric_weighted_score={rubric_weighted_score:.3f}."
    )
    return ScorerResult(
        scorer="fmea",
        passed=passed,
        score=score,
        rationale=rationale,
        details=details,
    )


def install(name: str = "fmea") -> None:
    """Register the deterministic FMEA scorer."""
    register(name, fmea_score)
