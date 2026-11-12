"""
watsonx.ai text generation used to build DiagnosticIQ Verbose.

Requires the ``dataset-creation`` dependency group and the ``WATSONX_APIKEY``,
``WATSONX_URL`` and ``WATSONX_PROJECT_ID`` environment variables.
"""

import os
from typing import Any, Dict, List, Optional

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as TextParams


def get_completion_response(
    message: str,
    model_id: str = "mistralai/mistral-large",
    max_tokens: int = 2000,
    temperature: float = 0,
    stop: Optional[List[str] | str] = None,
    seed: int = 20,
) -> Optional[Dict[str, Any] | List[Dict[str, Any]]]:
    if stop is None:
        stop = ["<>", "Note:"]
    elif isinstance(stop, str):
        stop = [stop]

    parameters = {
        TextParams.RANDOM_SEED: seed,
        TextParams.TEMPERATURE: temperature,
        TextParams.MAX_NEW_TOKENS: max_tokens,
        TextParams.MIN_NEW_TOKENS: 1,
        TextParams.STOP_SEQUENCES: stop,
    }

    credentials = Credentials(
        url=os.environ.get("WATSONX_URL", ""),
        api_key=os.environ.get("WATSONX_APIKEY", ""),
    )

    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=os.environ.get("WATSONX_PROJECT_ID", ""),
    )

    try:
        return model.generate(prompt=message)
    except Exception as ex:
        print(ex)
        return None


def clean_response(response: Optional[Dict[str, Any] | List[Dict[str, Any]]]) -> str:
    if not response:
        raise RuntimeError("Response is null!")

    if isinstance(response, list):
        response = response[0]

    return response["results"][0]["generated_text"]
