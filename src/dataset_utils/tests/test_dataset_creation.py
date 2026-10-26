import pickle
import sys

import pytest

from dataset_utils.config import DatasetCreationConfig
from dataset_utils.creation import create_dataset, load_dataset, save_dataset
from dataset_utils.outputs import question_formatting
from dataset_utils.paths import DATASET_DIR_ENV, get_dataset_dir
from dataset_utils.question import Question, QuestionDifficulty
from dataset_utils.reader import ADIQDataset
from dataset_utils.utils import file_handle
from dataset_utils.utils.tree import Node


def _rule(
    number: int, rule_id: str, asset: str, conditions: str, observations: list[str]
):
    return {
        "#n": number,
        "id": rule_id,
        "asset": asset,
        "display_text": {
            "date": "",
            "observations": observations,
            "rules": Node.generate_graph(conditions),
        },
    }


def test_create_save_and_load_dataset(tmp_path):
    rule_information = {
        "rule_set": [
            _rule(
                1,
                "RULE_1",
                "pump",
                "temperature is high OR pressure is low",
                ["bearing wear", "seal leak"],
            ),
            _rule(
                2,
                "RULE_2",
                "fan",
                "vibration is high OR airflow is low",
                ["blade imbalance", "filter blockage"],
            ),
        ]
    }
    observations = {
        observation: observation
        for observation in (
            "bearing wear",
            "seal leak",
            "blade imbalance",
            "filter blockage",
        )
    }
    similarities = {
        "RULE_1": ["RULE_2", "RULE_1"],
        "RULE_2": ["RULE_1", "RULE_2"],
    }
    config = DatasetCreationConfig(
        {
            "num_options": 2,
            "sel_random_sample": 1,
            "elem_random_sample": 1,
            "n_least_sim_rules": 2,
        }
    )

    dataset = create_dataset(
        rule_information,
        observations,
        {"pump": "A test pump", "fan": "A test fan"},
        similarities,
        config,
    )

    assert dataset["questions"]
    assert dataset["creation_config"]["num_options"] == 2
    assert all(
        len(question["options"])
        == len(question["option_ids"])
        == len(question["correct"])
        for question in dataset["questions"]
    )

    save_dataset(dataset, "smoke", str(tmp_path))
    loaded = load_dataset(str(tmp_path / "smoke"))
    reader = ADIQDataset(str(tmp_path / "smoke"))

    assert loaded["creation_config"] == dataset["creation_config"]
    assert len(reader.get_questions()) == len(dataset["questions"])


def test_reader_saves_path_locations_relative_to_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(DATASET_DIR_ENV, str(tmp_path))
    datasets_dir = tmp_path / "datasets"
    save_dataset({"questions": [], "date": ""}, "smoke", datasets_dir)

    reader = ADIQDataset(datasets_dir / "smoke")
    reader.save("smoke_copy", datasets_dir)

    metadata = file_handle.load_json(datasets_dir / "smoke_copy" / "metadata.json")
    assert metadata["loc"] == "datasets/smoke"


def test_dataset_dir_uses_explicit_path_or_environment(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    explicit = tmp_path / "explicit"
    monkeypatch.setenv(DATASET_DIR_ENV, str(configured))

    assert get_dataset_dir() == configured.resolve()
    assert get_dataset_dir(explicit) == explicit.resolve()


def test_dataset_dir_is_required(monkeypatch):
    monkeypatch.delenv(DATASET_DIR_ENV, raising=False)

    with pytest.raises(RuntimeError, match=DATASET_DIR_ENV):
        get_dataset_dir()


def test_load_pickle_remaps_legacy_node_module(tmp_path):
    node = Node.generate_graph("temperature is high OR pressure is low")
    payload = pickle.dumps(node, protocol=0)
    legacy_payload = payload.replace(b"dataset_utils.utils.tree", b"utils.tree")
    assert legacy_payload != payload

    path = tmp_path / "legacy-tree.pkl"
    path.write_bytes(legacy_payload)

    loaded = file_handle.load_pickle(path)

    assert isinstance(loaded, Node)
    assert loaded.statement == node.statement


def test_load_pickle_remaps_inline_main_node(monkeypatch, tmp_path):
    class InlineNode(Node):
        pass

    # Mimic a Node class defined inside a notebook, as in the rule-formalizing notebook.
    InlineNode.__module__ = "__main__"
    InlineNode.__qualname__ = InlineNode.__name__ = "Node"
    monkeypatch.setattr(sys.modules["__main__"], "Node", InlineNode, raising=False)
    node = InlineNode("AND", "temperature is high AND pressure is low")
    payload = pickle.dumps(node)
    monkeypatch.delattr(sys.modules["__main__"], "Node")
    assert b"__main__" in payload

    path = tmp_path / "inline-tree.pkl"
    path.write_bytes(payload)

    loaded = file_handle.load_pickle(path)

    assert type(loaded) is Node
    assert [c.statement for c in loaded.children] == [
        c.statement for c in node.children
    ]


def _question(**overrides) -> Question:
    fields = {
        "subject": "diagnosis_sense_simple_question_analysis",
        "id": 0,
        "question_id": "ADIQ_0",
        "rule_id": 1,
        "rule_name": "RULE_1",
        "date_detected": "",
        "asset_type": "pump",
        "condition_description": ["temperature is high", "pressure is low"],
        "temporal_condition": ["Met for 2 Hours"],
        "question": None,
        "question_prompt": "What should the operator look for?",
        "options": ["bearing wear", "seal leak"],
        "option_ids": ["A", "B"],
        "question_type": "positive",
        "correct": [True, False],
        "answer_str": "bearing wear",
        "rationale": "",
        "tags": [],
        "difficulty": QuestionDifficulty.entry,
    }
    fields.update(overrides)
    return Question(**fields)


def test_question_formatting_sets_final_prompt_fields():
    q = _question()

    question_formatting(q, {"pump": "A test pump"})

    assert q.question_first is True
    assert q.text_type == "choice"
    assert q.question == (
        "\n## Asset Description:\npump: A test pump\n\n"
        "## Conditions:\n- temperature is high\n- pressure is low\n\n"
        "## How long the conditions were met:\nMet for 2 Hours\n\n"
        "What should the operator look for?\n"
    )


def test_question_formatting_uses_verbalized_conditions_and_defaults():
    q = _question(temporal_condition=[], asset_type="fan")

    question_formatting(q, {}, condition_str="The temperature is high.")

    assert "## Conditions:\nThe temperature is high.\n" in q.question
    assert "fan: NONE" in q.question
    assert "## How long the conditions were met:\nNone\n" in q.question
