from evaluation.scorers.static_json import (
    evaluate_static_json,
    evaluate_static_json_batch,
    flatten_answer,
    parse_structured_answer,
)

def test_parse_json_object_from_noisy_markdown_answer():
    raw = 'Answer:\n```json\n{"energy": 3, "material": 12}\n```'

    assert parse_structured_answer(raw) == {"energy": 3, "material": 12}


def test_parse_fenced_json_response_key_from_noisy_answer():
    raw = (
        "Based on my analysis, here are the findings.\n\n"
        "```json\n"
        '{"response": "PMP42144 has 4 recurring air conditioner issues."}\n'
        "```"
    )

    assert parse_structured_answer(raw) == {
        "response": "PMP42144 has 4 recurring air conditioner issues."
    }


def test_parse_json_after_parenthetical_prose():
    raw = (
        'The registry has generic descriptions (all descriptions are "Asset PMPxxxxx").\n\n'
        '{"clarification": "Which pump do you mean by the big pump out the back?"}'
    )

    assert parse_structured_answer(raw) == {
        "clarification": "Which pump do you mean by the big pump out the back?"
    }


def test_parse_json_after_parenthetical_with_number():
    raw = (
        'Based on the repair history (asset PMP42144 has 17 work orders), '
        'the answer is:\n'
        '{"clarification": "Which asset do you mean by the main unit?"}'
    )

    assert parse_structured_answer(raw) == {
        "clarification": "Which asset do you mean by the main unit?"
    }


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


def test_parse_noisy_count_answer_prefers_final_standalone_number():
    raw = (
        "I checked the work orders and found no normal-operation jobs "
        "(such as tramming or normal machine movement) that should be counted.\n\n"
        "0"
    )

    assert parse_structured_answer(raw) == 0


def test_count_answer_does_not_treat_failed_step_number_as_the_answer():
    model_answer = "The final count cannot be provided due to the failure in Step 1."

    assert parse_structured_answer(model_answer) == model_answer
    score = evaluate_static_json("1", model_answer)
    assert score.strict_exact_match_accuracy == 0.0


def test_count_answer_does_not_extract_an_unlabelled_number_from_prose():
    model_answer = "The tool failed after examining 34 records."

    assert parse_structured_answer(model_answer) == model_answer
    score = evaluate_static_json("34", model_answer)
    assert score.strict_exact_match_accuracy == 0.0


def test_count_answer_compares_final_number_not_parenthetical_text():
    score = evaluate_static_json(
        "0",
        (
            "I checked the work orders and found no normal-operation jobs "
            "(such as tramming or normal machine movement) that should be counted.\n\n"
            "0"
        ),
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.f1 == 1.0
    assert score.details[0].model_value == "0"


def test_count_answer_with_wrong_final_number_fails_against_final_number():
    score = evaluate_static_json(
        "52",
        (
            'One work order mentions both "engine" and "motor"; this concerns '
            "an engine (the engine fan), so it counts as an engine job.\n\n"
            "55"
        ),
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.details[0].model_value == "55"


def test_parse_fault_code_as_categorical_string():
    assert parse_structured_answer("FC101") == "FC101"


def test_parse_json_object_after_parenthetical_text():
    raw = 'The dataset is ambiguous (no asset is specified).\n\n{"clarification": "Which asset?"}'

    assert parse_structured_answer(raw) == {"clarification": "Which asset?"}


def test_choice_answer_accepts_final_standalone_letter_after_explanation():
    score = evaluate_static_json(
        "C",
        (
            "I compared the excavator repair costs and C has the highest total.\n\n"
            "C"
        ),
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.partial_exact_match_accuracy == 1.0
    assert score.details[0].model_value == "c"


def test_choice_answer_accepts_final_answer_label():
    score = evaluate_static_json(
        "C",
        "The data points to excavator C.\n\nFinal Answer: C",
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.f1 == 1.0


def test_choice_answer_accepts_trailing_option_phrase():
    score = evaluate_static_json(
        "C",
        "After reviewing the options, the correct option is C.",
    )

    assert score.strict_exact_match_accuracy == 1.0


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


def test_anomaly_segment_scores_exact_categorical_and_numeric_delta():
    gold = {
        "condition": "faulty",
        "start_point": "240",
        "end_point": "511",
        "fault_type": "FC101",
    }
    model = {
        "condition": "faulty",
        "start_point": "241",
        "end_point": "512",
        "fault_type": "FC101",
    }

    score = evaluate_static_json(gold, model)
    details = {item.key: item for item in score.details}

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_exact_match_accuracy == 0.5
    assert score.partial_match_accuracy == 1.0
    assert score.partial_numeric_match_accuracy == 1.0
    assert score.range_match_accuracy == 0.5
    assert score.delta_1_match_accuracy == 1.0
    assert details["answer.condition"].match_type == "exact"
    assert details["answer.fault_type"].match_type == "exact"
    assert details["answer.start_point"].match_type == "partial_delta_1"
    assert details["answer.start_point"].range_match is True
    assert details["answer.start_point"].delta_1_match is True
    assert details["answer.end_point"].match_type == "partial_delta_1"
    assert details["answer.end_point"].range_match is False
    assert details["answer.end_point"].delta_1_match is True


def test_anomaly_segment_scores_numeric_range_match():
    gold = {
        "condition": "faulty",
        "start_point": "240",
        "end_point": "511",
        "fault_type": "FC101",
    }
    model = {
        "condition": "faulty",
        "start_point": "300",
        "end_point": "500",
        "fault_type": "FC101",
    }

    score = evaluate_static_json(gold, model)
    details = {item.key: item for item in score.details}

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_exact_match_accuracy == 0.5
    assert score.partial_match_accuracy == 1.0
    assert score.range_match_accuracy == 1.0
    assert score.delta_1_match_accuracy == 0.0
    assert details["answer.start_point"].match_type == "partial_range"
    assert details["answer.end_point"].match_type == "partial_range"


def test_count_only_delta_one_is_numeric_partial_match():
    score = evaluate_static_json("34", "35")

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_exact_match_accuracy == 0.0
    assert score.partial_match_accuracy == 1.0
    assert score.partial_numeric_match_accuracy == 1.0
    assert score.delta_1_match_accuracy == 1.0
    assert score.details[0].match_type == "partial_delta_1"


def test_numeric_answer_can_match_explicit_ground_truth_range():
    score = evaluate_static_json({"count": "10-12"}, {"count": 11})

    assert score.strict_exact_match_accuracy == 0.0
    assert score.partial_match_accuracy == 1.0
    assert score.partial_numeric_match_accuracy == 1.0
    assert score.range_match_accuracy == 1.0
    assert score.details[0].match_type == "partial_range"


def test_mode_clarification_accepts_equivalent_question_with_required_phrase():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {"clarification": "Can you clarify which asset you mean by the main unit?"},
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.mode_key_match == 1.0
    assert score.mode_exactly_one_key == 1.0
    assert score.mode_required_terms == ["main unit"]
    assert score.mode_matched_terms == ["main unit"]
    assert score.mode_term_coverage == 1.0


def test_mode_clarification_fails_when_mode_key_is_wrong():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the usual suspect'?"},
        {"response": "The usual suspect is pump PMP42144."},
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.mode_key_match == 0.0
    assert score.missing_keys == ["answer.clarification"]
    assert score.extra_keys == ["answer.response"]


def test_mode_abstain_requires_lately_dataset_date_time_terms():
    score = evaluate_static_json(
        {
            "abstain": (
                "Cannot determine what has been giving trouble 'lately' because "
                "the dataset does not contain date or time information."
            )
        },
        {
            "abstain": (
                "Cannot determine the lately-troublesome asset because the "
                "dataset has no date or time information."
            )
        },
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.mode_required_terms == [
        "lately",
        "cannot determine",
        "date",
        "dataset",
        "time",
    ]
    assert score.mode_term_coverage == 1.0


def test_mode_response_checks_asset_id_and_domain_terms():
    score = evaluate_static_json(
        {
            "response": (
                "No - the 'PMP' tag is unreliable. Tag PMP42144's work orders "
                "describe an air conditioner, steering/stick cylinders, a "
                "cracked handrail, tyres, a pressure vessel and an exhaust leak, "
                "none of which is pump work, so the tag does not indicate a pump."
            )
        },
        {
            "response": (
                "No. PMP42144 should not be treated as a pump; the tag is "
                "unreliable because the history points to air conditioner, "
                "steering, handrail, pressure vessel, and exhaust leak work."
            )
        },
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert "pmp42144" in score.mode_required_terms
    assert "pump" in score.mode_required_terms
    assert "unreliable" in score.mode_required_terms
    assert score.mode_term_coverage == 1.0


def test_mode_requires_exactly_one_top_level_key():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {
            "clarification": "Which asset do you mean by the main unit?",
            "response": "I can also answer once you clarify.",
        },
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.mode_key_match == 0.0
    assert score.mode_exactly_one_key == 0.0
    assert score.extra_keys == ["answer.clarification", "answer.response"]


def test_car_metadata_passes_on_mode_and_required_terms_only():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {"clarification": "Which unit are you referring to as the main unit?"},
        evaluation_metadata={
            "mode": "clarification",
            "required_terms": ["main unit"],
            "optional_terms": ["asset"],
        },
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.car_score == 1.0
    assert score.mode_key_match == 1.0
    assert score.mode_required_terms == ["main unit"]
    assert score.mode_matched_terms == ["main unit"]
    assert score.mode_optional_terms == ["asset"]
    assert score.mode_matched_optional_terms == []
    assert score.mode_optional_term_coverage == 0.0


def test_car_optional_terms_do_not_inflate_similarity():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {
            "clarification": (
                "Which asset do you mean by the main unit? Please provide the "
                "asset tag."
            )
        },
        evaluation_metadata={
            "mode": "clarification",
            "required_terms": ["main unit"],
            "optional_terms": ["asset", "asset tag"],
        },
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.partial_similarity_score == 1.0
    assert score.mode_optional_term_coverage == 1.0


def test_car_metadata_passes_when_any_required_term_matches():
    score = evaluate_static_json(
        {"abstain": "Cannot determine because date or time is missing."},
        {"abstain": "Cannot determine this from the available date fields."},
        evaluation_metadata={
            "mode": "abstain",
            "required_terms": ["lately", "date", "time"],
        },
    )

    assert score.strict_exact_match_accuracy == 1.0
    assert score.car_score == 0.7333333333333333
    assert score.mode_key_match == 1.0
    assert score.mode_matched_terms == ["date"]
    assert score.mode_term_coverage == 1 / 3


def test_car_metadata_fails_when_required_term_is_missing():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {"clarification": "Which asset should I analyze?"},
        evaluation_metadata={
            "mode": "clarification",
            "required_terms": ["main unit"],
        },
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.car_score == 0.6
    assert score.mode_key_match == 1.0
    assert score.mode_term_coverage == 0.0


def test_car_metadata_fails_when_mode_key_is_wrong_even_with_required_term():
    score = evaluate_static_json(
        {"clarification": "Which asset do you mean by 'the main unit'?"},
        {"response": "The main unit appears to be PMP42144."},
        evaluation_metadata={
            "mode": "clarification",
            "required_terms": ["main unit"],
        },
    )

    assert score.strict_exact_match_accuracy == 0.0
    assert score.car_score == 0.4
    assert score.mode_key_match == 0.0
    assert score.mode_term_coverage == 1.0


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



from evaluation.models import Scenario
from evaluation.scorers.static_json import StaticJsonScorer


def test_static_json_scorer_wrapper_exact_match():
    scenario = Scenario.from_raw(
        {
            "id": "11",
            "text": "Count storage jobs.",
            "expected_answer": "{'energy': 14, 'material': 48}",
            "scoring_method": "static_json",
        }
    )

    scorer = StaticJsonScorer()
    result = scorer(
        scenario,
        '{"energy": 14, "material": 48}',
        "",
    )

    assert result.scorer == "static_json"
    assert result.passed is True
    assert result.score == 1.0
    assert result.details["strict_exact_match_accuracy"] == 1.0


def test_static_json_scorer_uses_car_metadata_score():
    scenario = Scenario.from_raw(
        {
            "id": "151",
            "text": "Clarify main unit.",
            "expected_answer": '{"clarification": "Which asset do you mean by the main unit?"}',
            "evaluation_metadata": {
                "mode": "clarification",
                "required_terms": ["main unit"],
            },
            "scoring_method": "static_json",
        }
    )

    scorer = StaticJsonScorer()
    result = scorer(
        scenario,
        '{"clarification": "Which asset is the main unit?"}',
        "",
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["car_score"] == 1.0
