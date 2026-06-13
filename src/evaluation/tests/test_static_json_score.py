from evaluation.static_json_score import (
    evaluate_static_json,
    evaluate_static_json_batch,
    flatten_answer,
    parse_structured_answer,
)


def test_parse_json_object_from_noisy_markdown_answer():
    raw = 'Answer:\n```json\n{"energy": 3, "material": 12}\n```'

    assert parse_structured_answer(raw) == {"energy": 3, "material": 12}


def test_parse_python_style_dict():
    raw = "{'energy': 14, 'material': 48}"

    assert parse_structured_answer(raw) == {"energy": 14, "material": 48}


def test_parse_python_style_list_of_tuples():
    raw = '[("Engines & motors", 5), ("Lines & drives", 2)]'

    assert parse_structured_answer(raw) == [
        ("Engines & motors", 5),
        ("Lines & drives", 2),
    ]


def test_parse_count_only_answer():
    assert parse_structured_answer("34") == 34


def test_parse_noisy_count_answer():
    assert parse_structured_answer("The answer is 34.") == 34


def test_flatten_nested_json():
    answer = {"a": {"b": 2}, "c": [3, {"d": 4}]}

    assert flatten_answer(answer) == {
        "answer.a.b": "2",
        "answer.c[0]": "3",
        "answer.c[1].d": "4",
    }


def test_flatten_tuple_list_answer():
    answer = '[("Engines & motors", 5), ("Lines & drives", 2)]'

    assert flatten_answer(answer) == {
        "answer[0][0]": "engines & motors",
        "answer[0][1]": "5",
        "answer[1][0]": "lines & drives",
        "answer[1][1]": "2",
    }


def test_exact_match_json_object_with_prefix():
    score = evaluate_static_json(
        {"energy": 3, "material": 12},
        'Final Answer: {"energy": 3, "material": 12}',
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.partial_exact_match_accuracy == 1.0
    assert score.partial_similarity_score == 1.0
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0
    assert score.missing_keys == []
    assert score.extra_keys == []


def test_missing_and_extra_keys_are_reported():
    score = evaluate_static_json(
        {"energy": 3, "material": 12},
        {"energy": 3, "other": 99},
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.exact_value_matches == 1
    assert score.missing_keys == ["answer.material"]
    assert score.extra_keys == ["answer.other"]


def test_wrong_value_is_not_strict_match():
    score = evaluate_static_json(
        {"energy": 14, "material": 48},
        {"energy": 14, "material": 27},
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_exact_match_accuracy == 0.5
    assert score.exact_value_matches == 1
    assert score.missing_keys == []
    assert score.extra_keys == []


def test_numeric_partial_similarity():
    score = evaluate_static_json({"count": 100}, {"count": 104})

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_similarity_score == 0.7


def test_count_only_exact_match():
    score = evaluate_static_json("34", "The answer is 34.")

    assert score.strict_exact_match_accuracy == 1.0
    assert score.f1 == 1.0


def test_batch_evaluation():
    result = evaluate_static_json_batch(
        [
            ({"energy": 3}, {"energy": 3}),
            ({"material": 10}, {"material": 9}),
        ]
    )

    assert result["num_examples"] == 2
    assert result["strict_exact_match_accuracy"] == 0.5