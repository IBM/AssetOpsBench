"""Minimal Industrial Automation Challenge predictor entry point."""

from __future__ import annotations


def predict(scenario):
    """Return a placeholder MCQA answer.

    Replace this function with your model/agent call. For Kaggle submissions the
    important scored field is ``answer``: a single option letter such as A/B/C.
    Optional reasoning and trajectory fields are kept for offline audit packages.
    """

    return {
        "answer": "A",
        "prediction": "A",
        "reasoning": "Bundled baseline predictor always selects option A.",
        "trajectory": [],
    }
