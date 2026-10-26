import copy
import string
from enum import Enum
from typing import Any, Dict


class QuestionTemplateMode(Enum):
    random = "random"
    pre_select = "pre-select"


class DatasetCreationConfig:
    # default values
    subject = "diagnosis_sense_simple_question_analysis"
    db_acronym = "ADIQ"
    db_name = "AssetDiagnosisIQ"
    option_choices_string: str = string.ascii_uppercase
    question_first = None
    text_type = "choice"
    relevancy = None
    num_options = 3
    question_template_mode: QuestionTemplateMode = QuestionTemplateMode.pre_select
    question_template_pre_select_options = [0]
    question_template_random_count = 1
    sel_random_sample = 10
    elem_random_sample = 25
    n_least_sim_rules = 25
    question_condition_template = """
## Asset Description:
{asset_type}: {asset_description}

## Conditions:
{conditions}

{question_prompt}
"""

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        for k, v in config_dict.items():
            setattr(self, k, v)

    def to_config(self) -> Dict[str, Any]:
        config = {}

        # Collect all attributes from class defaults
        for attr, value in vars(self.__class__).items():
            if not attr.startswith("_") and not callable(value):
                config[attr] = getattr(self, attr)

        # Collect any attributes that were added dynamically to the instance
        for attr, value in self.__dict__.items():
            if not attr.startswith("_") and attr not in config:
                config[attr] = value

        # Convert Enums to their value representation
        for k, v in config.items():
            if isinstance(v, Enum):
                config[k] = v.value

        return config

    @staticmethod
    def from_dict(_dict: Dict[str, str]) -> Any:
        config_dict: Dict[str, Any] = copy.deepcopy(_dict)
        config_dict["question_template_mode"] = QuestionTemplateMode(
            config_dict["question_template_mode"]
        )

        return DatasetCreationConfig(config_dict)
