"""
Any conversion of a question object to any output format is considered here.
"""

from typing import Dict, Optional

from .question import Question

QUESTION_TEMPLATE = """
## Asset Description:
{asset_type}: {asset_desc}

## Conditions:
{condition_str}

## How long the conditions were met:
{temporal_condition}

{question_prompt}
"""


def question_templating(
    q: Question, desc_dict: Dict[str, str], condition_str: Optional[str] = None
) -> str:
    if condition_str is None:
        condition_str = "\n".join(
            list(map(lambda x: "- " + x, q.condition_description))
        )
    temporal_condition = (
        q.temporal_condition[0] if len(q.temporal_condition) > 0 else "None"
    )
    return QUESTION_TEMPLATE.format(
        asset_type=q.asset_type,
        asset_desc=desc_dict.get(q.asset_type, "NONE"),
        condition_str=condition_str,
        temporal_condition=temporal_condition,
        question_prompt=q.question_prompt,
    )


def question_formatting(
    q: Question, desc_dict: Dict[str, str], condition_str: Optional[str] = None
) -> None:
    """Write the final question text and prompt flags onto ``q``.

    This is the post-creation step that formatted the released DiagnosticIQ
    questions. ``condition_str`` replaces the symbolic condition list, e.g.
    with ``q.verberlized_conditions`` for DiagnosticIQ Verbose.
    """
    q.question_first = True
    q.text_type = "choice"
    q.question = question_templating(q, desc_dict, condition_str)


def to_basic_prompt(
    q: Question, template: str, asset_descriptions: Dict[str, str]
) -> str:
    return template.format(
        asset_type=q.asset_type,
        asset_description=asset_descriptions.get(q.asset_type, "NONE"),
        conditions="\n".join(list(map(lambda x: "- " + x, q.condition_description))),
        temporal_condition=q.temporal_condition[0]
        if len(q.temporal_condition) > 0
        else "NONE",
        question_prompt=q.question_prompt,
        options="\n".join(
            ["{}). {}".format(op_id, op) for op_id, op in zip(q.option_ids, q.options)]
        ),
    )
