from __future__ import annotations

import pandas as pd
import pytest

from competition.metric import score


def test_metric_scores_exact_answer_accuracy():
    solution = pd.DataFrame(
        {
            "id": ["toy-q1", "toy-q2", "toy-q3"],
            "answer": ["A", "B", "C"],
        }
    )
    submission = pd.DataFrame(
        {
            "id": ["toy-q1", "toy-q2", "toy-q3"],
            "answer": ["A", "c", "C"],
        }
    )

    assert score(solution, submission) == pytest.approx(2 / 3)


def test_metric_requires_answer_column():
    solution = pd.DataFrame({"id": ["toy-q1"], "answer": ["A"]})
    submission = pd.DataFrame({"id": ["toy-q1"], "prediction": ["A"]})

    with pytest.raises(ValueError, match="submission is missing required column"):
        score(solution, submission)


def test_metric_rejects_missing_prediction_rows():
    solution = pd.DataFrame({"id": ["toy-q1", "toy-q2"], "answer": ["A", "B"]})
    submission = pd.DataFrame({"id": ["toy-q1"], "answer": ["A"]})

    with pytest.raises(ValueError, match="missing predictions"):
        score(solution, submission)
