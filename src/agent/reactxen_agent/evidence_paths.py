"""Normalize the exact list wrapper some model tool calls emit."""

from typing import Annotated

from pydantic import BeforeValidator, StringConstraints


def unwrap_item_list(value):
    # Accept only this unambiguous wrapper. Other dictionaries/scalars remain
    # invalid, and list entries still undergo strict string validation.
    if (
        isinstance(value, dict)
        and set(value) == {"item"}
        and isinstance(value["item"], list)
    ):
        return value["item"]
    return value


EvidencePath = Annotated[str, StringConstraints(strict=True, min_length=1)]
EvidencePaths = Annotated[list[EvidencePath], BeforeValidator(unwrap_item_list)]
