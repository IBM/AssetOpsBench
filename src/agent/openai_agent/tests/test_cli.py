"""Tests for the OpenAI agent CLI."""

from agent.openai_agent.cli import _build_parser


def test_help_documents_tokenrouter_model_ids():
    assert "tokenrouter/<model>" in _build_parser().format_help()
