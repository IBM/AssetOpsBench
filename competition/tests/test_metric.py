from __future__ import annotations

import pandas as pd
import pytest

from competition.metric import public_private_accuracy, score


def test_metric_scores_exact_answer_accuracy():
    solution = pd.DataFrame(
        {
            "id": ["q1", "q2", "q3"],
            "answer": ["A", "B", "C"],
            "usage": ["Public", "Private", "Private"],
        }
    )
    submission = pd.DataFrame(
        {
            "id": ["q1", "q2", "q3"],
            "answer": ["A", "c", "C"],
        }
    )

    assert score(solution, submission) == pytest.approx(2 / 3)


def test_metric_reports_public_private_accuracy():
    solution = pd.DataFrame(
        {
            "id": ["q1", "q2", "q3", "q4"],
            "answer": ["A", "B", "C", "D"],
            "usage": ["Public", "Public", "Private", "Private"],
        }
    )
    submission = pd.DataFrame(
        {
            "id": ["q1", "q2", "q3", "q4"],
            "answer": ["A", "A", "C", "D"],
        }
    )

    assert public_private_accuracy(solution, submission) == {
        "overall": pytest.approx(0.75),
        "public": pytest.approx(0.5),
        "private": pytest.approx(1.0),
    }


def test_metric_requires_answer_column():
    solution = pd.DataFrame({"id": ["q1"], "answer": ["A"]})
    submission = pd.DataFrame({"id": ["q1"], "prediction": ["A"]})

    with pytest.raises(ValueError, match="submission is missing required column"):
        score(solution, submission)
