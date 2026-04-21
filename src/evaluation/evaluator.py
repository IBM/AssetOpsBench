import json
import re
from functools import lru_cache
from pathlib import Path


IOT_DATA_PATH = Path(
    "src/couchdb/sample_data/iot/chiller6_june2020_sensordata_couchdb.json"
)


def load_questions(file_path: str = "src/evaluation/questions.json") -> list[dict]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s/.\-%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def looks_like_file_reference(answer: str) -> bool:
    lowered = answer.lower()
    file_signals = [".json", ".csv", ".txt", ".pdf", "file", "saved", "attached", "/"]
    return any(signal in lowered for signal in file_signals)


@lru_cache(maxsize=1)
def load_iot_documents() -> list[dict]:
    if not IOT_DATA_PATH.exists():
        return []

    with IOT_DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data if isinstance(data, list) else []


@lru_cache(maxsize=1)
def get_known_sites() -> tuple[str, ...]:
    return ("MAIN",)


@lru_cache(maxsize=1)
def get_known_assets() -> tuple[str, ...]:
    assets = {
        document["asset_id"]
        for document in load_iot_documents()
        if document.get("asset_id")
    }
    return tuple(sorted(assets))


@lru_cache(maxsize=None)
def get_known_sensors(asset_id: str) -> tuple[str, ...]:
    excluded_fields = {"_id", "_rev", "asset_id", "timestamp"}
    for document in load_iot_documents():
        if document.get("asset_id") == asset_id:
            sensors = sorted(key for key in document if key not in excluded_fields)
            return tuple(sensors)
    return ()


def entity_aliases(entity: str) -> set[str]:
    normalized = normalize_text(entity)
    aliases = {normalized}

    chiller_match = re.fullmatch(r"chiller\s+(\d+)", normalized)
    if chiller_match:
        aliases.add(f"chiller {chiller_match.group(1)}")

    sensor_prefix = re.match(r"chiller\s+\d+\s+(.+)", normalized)
    if sensor_prefix:
        aliases.add(sensor_prefix.group(1))

    return {alias for alias in aliases if alias}


def answer_contains_entity(answer: str, entity: str) -> bool:
    normalized_answer = normalize_text(answer)
    return any(alias in normalized_answer for alias in entity_aliases(entity))


def find_expected_mentions(answer: str, expected_entities: list[str]) -> list[str]:
    return [
        entity
        for entity in expected_entities
        if answer_contains_entity(answer, entity)
    ]


def find_chiller_mentions(answer: str) -> set[str]:
    return {
        f"Chiller {match}"
        for match in re.findall(r"\bchiller\s*(\d+)\b", answer, flags=re.IGNORECASE)
    }


def find_unexpected_assets(answer: str, expected_assets: list[str]) -> list[str]:
    expected_normalized = {normalize_text(asset) for asset in expected_assets}
    unexpected = [
        asset
        for asset in find_chiller_mentions(answer)
        if normalize_text(asset) not in expected_normalized
    ]
    return sorted(unexpected)


def find_unexpected_sites(answer: str, expected_sites: list[str]) -> list[str]:
    expected_normalized = {normalize_text(site) for site in expected_sites}
    known_site_tokens = {"MAIN", "PLANT_A"}
    unexpected = [
        site
        for site in known_site_tokens
        if site.lower() in answer.lower()
        and normalize_text(site) not in expected_normalized
    ]
    return sorted(unexpected)


def find_unexpected_sensors(
    answer: str,
    expected_sensors: list[str],
    matched_sensors: list[str],
) -> list[str]:
    normalized_answer = normalize_text(answer)
    matched_aliases = set()
    for sensor in matched_sensors:
        matched_aliases.update(entity_aliases(sensor))

    common_sensor_terms = [
        "temperature",
        "pressure",
        "flow",
        "tonnage",
        "efficiency",
        "schedule",
        "power input",
        "run status",
        "humidity",
        "vibration",
    ]
    unexpected = []

    for term in common_sensor_terms:
        if term in normalized_answer and not any(
            term in alias for alias in matched_aliases
        ):
            unexpected.append(term)

    if "sensor" in normalized_answer and not matched_sensors:
        unexpected.append("unspecified sensor")

    return sorted(set(unexpected))


def score_entities(
    answer: str,
    expected_entities: list[str],
    unexpected_entities: list[str],
    allow_file_reference: bool,
) -> dict:
    matched = find_expected_mentions(answer, expected_entities)
    missing = [entity for entity in expected_entities if entity not in matched]

    expected_count = len(expected_entities)
    matched_count = len(matched)
    unexpected_count = len(unexpected_entities)

    recall = matched_count / expected_count if expected_count else 0.0
    precision_denominator = matched_count + unexpected_count
    precision = (
        matched_count / precision_denominator
        if precision_denominator
        else 0.0
    )

    score = (0.7 * recall) + (0.3 * precision)
    file_reference = looks_like_file_reference(answer)
    if allow_file_reference and file_reference:
        score = max(score, 0.35)

    return {
        "score": round(min(score, 1.0), 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "expected_count": expected_count,
        "matched_count": matched_count,
        "unexpected_count": unexpected_count,
        "matched_expected": matched,
        "missing_expected": missing,
        "unexpected_entities": unexpected_entities,
        "file_reference": file_reference,
    }


def evaluate_asset_list(answer: str, question_data: dict) -> dict:
    expected_assets = list(get_known_assets())
    unexpected_assets = find_unexpected_assets(answer, expected_assets)
    unexpected_sites = find_unexpected_sites(answer, [question_data.get("site_name", "MAIN")])

    return score_entities(
        answer=answer,
        expected_entities=expected_assets,
        unexpected_entities=unexpected_assets + unexpected_sites,
        allow_file_reference=question_data.get("allow_file_reference", False),
    )


def evaluate_sensor_list(answer: str, question_data: dict) -> dict:
    asset_id = question_data.get("asset_id", "")
    expected_sensors = list(get_known_sensors(asset_id))
    matched_sensors = find_expected_mentions(answer, expected_sensors)
    unexpected_sensors = find_unexpected_sensors(
        answer,
        expected_sensors,
        matched_sensors,
    )

    result = score_entities(
        answer=answer,
        expected_entities=expected_sensors,
        unexpected_entities=unexpected_sensors,
        allow_file_reference=question_data.get("allow_file_reference", False),
    )
    result["matched_expected"] = matched_sensors
    result["missing_expected"] = [
        sensor for sensor in expected_sensors if sensor not in matched_sensors
    ]
    return result


def evaluate_site_list(answer: str, question_data: dict) -> dict:
    expected_sites = list(get_known_sites())
    unexpected_sites = find_unexpected_sites(answer, expected_sites)

    return score_entities(
        answer=answer,
        expected_entities=expected_sites,
        unexpected_entities=unexpected_sites,
        allow_file_reference=question_data.get("allow_file_reference", False),
    )


def summarize_evaluation(details: dict) -> str:
    matched = details.get("matched_count", 0)
    expected = details.get("expected_count", 0)
    unexpected = details.get("unexpected_count", 0)
    summary = f"matched {matched}/{expected}; unexpected {unexpected}"

    missing = details.get("missing_expected", [])
    if missing:
        summary += f"; missing: {', '.join(missing[:2])}"
        if len(missing) > 2:
            summary += f" +{len(missing) - 2} more"

    unexpected_entities = details.get("unexpected_entities", [])
    if unexpected_entities:
        summary += f"; extra: {', '.join(unexpected_entities[:2])}"
        if len(unexpected_entities) > 2:
            summary += f" +{len(unexpected_entities) - 2} more"

    if details.get("file_reference"):
        summary += "; file ref"

    return summary


def evaluate_answer_details(answer: str, question_data: dict) -> dict:
    if not answer:
        return {
            "score": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "expected_count": 0,
            "matched_count": 0,
            "unexpected_count": 0,
            "matched_expected": [],
            "missing_expected": [],
            "unexpected_entities": [],
            "file_reference": False,
            "summary": "empty answer",
        }

    evaluation_type = question_data.get("evaluation_type")

    if evaluation_type == "asset_list":
        details = evaluate_asset_list(answer, question_data)
    elif evaluation_type == "sensor_list":
        details = evaluate_sensor_list(answer, question_data)
    elif evaluation_type == "site_list":
        details = evaluate_site_list(answer, question_data)
    else:
        required_keywords = question_data.get("required_keywords", [])
        details = score_entities(
            answer=answer,
            expected_entities=required_keywords,
            unexpected_entities=[],
            allow_file_reference=question_data.get("allow_file_reference", False),
        )

    details["summary"] = summarize_evaluation(details)
    return details


def evaluate_answer(answer: str, question_data: dict) -> float:
    return evaluate_answer_details(answer, question_data)["score"]
