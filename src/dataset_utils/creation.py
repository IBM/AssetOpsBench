import copy
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .config import DatasetCreationConfig, QuestionTemplateMode
from .question import Question, QuestionDifficulty
from .utils import file_handle
from .utils.tree import OPS, Node, get_tree_repr

selection_question_library = [
    "Given the above detected conditions, what should the operator look for?",
    "Analyse the given conditions of the presented asset and select the option that MOST likely gives the reason for the conditions?",
    "What is the MOST plausible explanation for the observed conditions of the asset?",
    "Review the listed conditions and identify which option MOST accurately accounts for them.",
    # Analytical / Diagnostic tone
    "Considering the conditions described, which option best diagnoses the likely cause?",
    "What underlying factor, as presented in the options, could explain the current asset state?",
    "Select the option that offers the MOST logical interpretation of the asset’s present conditions.",
    # Decision-oriented
    "If you were to advise the operator, which option should they prioritize, based on the conditions?",
    "From the given options, which one MOST likely corresponds to the observed asset behavior?",
    # Slightly Conversational
    "Given what is known about the asset’s conditions, which explanation makes the MOST sense?",
    "Looking at the current state of the asset, what is the MOST likely cause among the options?",
]

elimination_question_library = [
    "Given the above detected conditions, what should NOT the operator look for?",
    "Analyse the given conditions of the presented asset and select the option that LEAST likely gives the reason for the conditions?",
    "What is the LEAST plausible explanation for the observed conditions of the asset?",
    "Review the listed conditions and identify which option LEAST likely to account for them.",
    # Analytical / Diagnostic tone
    "Considering the conditions described, which option best diagnoses the LEAST likely cause?",
    "What underlying factor, as presented in the options, could NOT explain the current asset state?",
    "Select the option that offers the LEAST logical interpretation of the asset’s present conditions.",
    # Decision-oriented
    "If you were to advise the operator, which option should they ELIMINATE, based on the conditions?",
    "From the given options, which one LEAST likely corresponds to the observed asset behavior?",
    # Slightly Conversational
    "Given what is known about the asset’s conditions, which explanation makes the LEAST sense?",
    "Looking at the current state of the asset, what is the LEAST likely cause among the options?",
]


def save_dataset(dataset: dict[str, Any], dataset_name: str, loc: str) -> None:
    dir = os.path.join(loc, dataset_name)

    if os.path.exists(dir):
        print("Overridding Dataset")

    os.makedirs(dir, exist_ok=True)

    if "questions" not in dataset:
        raise RuntimeError("Question list is necessary")

    file_handle.save_jsonl(dataset["questions"], os.path.join(dir, "data.jsonl"))

    file_handle.save_json(
        {k: v for k, v in dataset.items() if k != "questions"},
        os.path.join(dir, "metadata.json"),
    )


def load_dataset(loc: str) -> Dict[str, Any]:

    metadata_path = os.path.join(loc, "metadata.json")
    data_path = os.path.join(loc, "data.jsonl")

    questions = file_handle.load_jsonl(data_path)
    data = file_handle.load_json(metadata_path)

    data["questions"] = questions

    return data


def create_question(
    info: dict[str, Any],
    uniq_obs: dict[str, str],
    qid: int,
    asset_description: Dict[str, str],
    rule_id_similarity_map: Dict[str, List[str]],
    rule_info: Dict[str, Dict[str, Any]],
    config: DatasetCreationConfig,
) -> List[Question]:

    questions = []

    observations_for_info = observations_selection(
        info, uniq_obs, rule_id_similarity_map, rule_info, config
    )

    conditions_for_info = condition_selection(info, config)

    for ind1, cinfo in enumerate(conditions_for_info):
        for ind2, oinfo in enumerate(observations_for_info):
            oinfo["question"] = config.question_condition_template.format(
                asset_type=info["asset"],
                asset_description=asset_description.get(info["asset"], "#*#*#"),
                conditions="\n".join(list(map(lambda x: "- " + x, cinfo))),
                question_prompt=oinfo["question_prompt"],
            )

            ques = Question(
                subject=config.subject,
                id=qid + ind1 * len(observations_for_info) + ind2,
                question_id="{db_name}_{qid}".format(
                    db_name=config.db_acronym,
                    qid=qid + ind1 * len(observations_for_info) + ind2,
                ),
                rule_id=info["#n"],
                rule_name=info["id"],
                date_detected=info["display_text"]["date"]
                if info["display_text"]["date"]
                else "",
                asset_type=info["asset"],
                **oinfo,
                condition_description=cinfo["non-temporal"],
                temporal_condition=cinfo["temporal"],
                rationale="",
                tags=[
                    f"{info['#n']}",
                    info["asset"],
                    QuestionDifficulty.entry.value,
                ],
                difficulty=QuestionDifficulty.entry,
            )

            questions.append(ques)

    return questions


def remove_brackets(s: str, start: str, end: str):
    stack = []
    result = list(s)
    to_remove = set()

    for i, char in enumerate(s):
        if char == start:
            stack.append(i)
        elif char == end:
            if stack:
                stack.pop()
            else:
                to_remove.add(i)

    to_remove.update(stack)
    return "".join([char for i, char in enumerate(result) if i not in to_remove])


def dangling_bracket_remove(
    sentences: List[str],
    keywords: List[Dict[str, str]] = [
        {"start": "{", "close": "}"},
        {"start": "[", "close": "]"},
        {"start": "(", "close": ")"},
    ],
) -> List[str]:

    removed: List[str] = []
    for sentence in sentences:
        for k in keywords:
            sentence = remove_brackets(sentence, k["start"], k["close"])

        removed.append(sentence)

    return removed


def traverse_tree(root: Node) -> Node:
    _temp = copy.deepcopy(root)
    recursive_traverse_tree(_temp)
    concat_tree(None, _temp)
    return _temp


def concat_tree(parent: Optional[Node], child: Node):
    if child.children:
        for c in child.children:
            concat_tree(child, c)

        if len(child.children) == 1 and parent:
            parent.children.remove(child)
            parent.children.append(child.children[0])


def recursive_traverse_tree(root: Node) -> None:
    if root.children:
        for c in root.children:
            recursive_traverse_tree(c)

        if root.val == OPS.OR:
            sel_ind = random.randint(0, len(root.children) - 1)
            root.children = [root.children[sel_ind]]
        else:
            root.children = root.children


def tree_2_string(n: Node) -> str:
    if not n.children:
        return n.statement if n.statement else "None"

    _s = ""
    for i, c in enumerate(n.children):
        if i > 0:
            _s += n.val if n.val else "None"

        _s += tree_2_string(c)

    return _s


def _get_condition_list(root: Node) -> List[str]:
    _s = []
    for c in root.children:
        _s.append(
            # tree_2_string(c).replace("(", "").replace(")", "").replace("\n", "").strip()
            tree_2_string(c).replace("\n", "").strip()
        )
    return _s


def get_all_possible_trees_random(root: Node, times: int = 10) -> List[List[str]]:
    trees = []
    for _ in range(times):
        trees.append(traverse_tree(root))

    list_trees = []
    for t in trees:
        list_trees.append(_get_condition_list(t))

    return list_trees


def find_unique_lists(ltrees: List[List[str]]) -> List[List[str]]:
    unq_ltrees = []

    while len(ltrees):
        val = ltrees.pop(0)
        unique = True
        for ut in unq_ltrees:
            if set(val) == set(ut):
                unique = False
                break

        if unique:
            unq_ltrees.append(val)

    return unq_ltrees


def get_list_conditions(n: Node) -> List[List[str]]:
    ltrees = get_all_possible_trees_random(n)
    ltrees = find_unique_lists(ltrees)
    ltrees = [dangling_bracket_remove(x) for x in ltrees]

    return ltrees


def condition_selection(
    info: dict[str, Any], config: DatasetCreationConfig
) -> List[Dict[str, List[str]]]:

    conds = get_list_conditions(info["display_text"]["rules"])

    condition_splits = []
    for lt in conds:
        _c = {"temporal": [], "non-temporal": []}

        for cond in lt:
            if "Met for " in cond:
                _c["temporal"].append(cond)
            else:
                _c["non-temporal"].append(cond)

        condition_splits.append(_c)

    return condition_splits


def observations_selection(
    info: dict[str, Any],
    uniq_obs: dict[str, str],
    rule_id_similarity_map: Dict[str, List[str]],
    rule_info: Dict[str, Dict[str, Any]],
    config: DatasetCreationConfig,
) -> List[Dict[str, Any]]:

    conditions = []

    if config.question_template_mode == QuestionTemplateMode.pre_select:
        ques_index = config.question_template_pre_select_options
    else:
        option_num = min(
            len(selection_question_library), len(elimination_question_library)
        )

        if option_num > config.question_template_random_count:
            ques_index = random.sample(
                list(range(option_num)), config.question_template_random_count
            )
        else:
            ques_index = list(range(option_num))
            random.shuffle(ques_index)

    for qindex in ques_index:
        for _ops in selection_options(
            info,
            uniq_obs,
            rule_id_similarity_map,
            rule_info,
            num_obs=config.num_options,
            options_string=config.option_choices_string,
            number_random_sample=config.sel_random_sample,
            n_least_sim_rules=config.n_least_sim_rules,
        ):
            _ops["question_prompt"] = selection_question_library[qindex]
            conditions.append(_ops)

        for _ops in elimination_options(
            info,
            uniq_obs,
            rule_id_similarity_map,
            rule_info,
            options_string=config.option_choices_string,
            number_random_sample=config.elem_random_sample,
            n_least_sim_rules=config.n_least_sim_rules,
        ):
            _ops["question_prompt"] = elimination_question_library[qindex]
            conditions.append(_ops)

    return conditions


def selection_options(
    info: dict[str, Any],
    uniq_obs: dict[str, str],
    rule_id_similarity_map: Dict[str, List[str]],
    rule_info: Dict[str, Dict[str, Any]],
    options_string: str,
    num_obs=3,
    variable_num_obs=False,
    number_random_sample=10,
    n_least_sim_rules=25,
) -> List[Dict[str, Any]]:

    if variable_num_obs:
        raise NotImplementedError()

    _temp = []
    rule_obs = [uniq_obs[x] for x in info["display_text"]["observations"]]
    sel_rules = rule_id_similarity_map[info["id"]][:n_least_sim_rules]

    _obs = set()
    for r in sel_rules:
        r = rule_info[r]["display_text"]["observations"]
        _obs = _obs.union(set([uniq_obs[x] for x in r]))

    # print("Used Rules {} and Observations {}".format(len(sel_rules), len(_obs)))
    for r in rule_obs:
        for _ in range(number_random_sample):
            possible_choices = list(set(_obs) - set(rule_obs))

            _temp.append(
                {
                    "answer": r,
                    "rule_choice": [r],
                    "other_choice": random.sample(possible_choices, num_obs - 1),
                }
            )

    cleaned = []
    for _t in _temp:
        all_options = _t["rule_choice"] + _t["other_choice"]
        ind = list(range(len(all_options)))
        random.shuffle(ind)
        ind = np.array(ind)
        answer_ind = ind == 0

        cleaned.append(
            {
                "answer_str": _t["answer"],
                "options": [all_options[x] for x in ind],
                "option_ids": [options_string[i] for i in range(len(all_options))],
                "question_type": "positive",
                "correct": answer_ind.tolist(),
            }
        )

    return cleaned


def elimination_options(
    info: dict[str, Any],
    uniq_obs: dict[str, str],
    rule_id_similarity_map: Dict[str, List[str]],
    rule_info: Dict[str, Dict[str, Any]],
    options_string: str,
    num_obs=3,
    variable_num_obs=False,
    number_random_sample=10,
    n_least_sim_rules=25,
) -> List[Dict[str, Any]]:

    if variable_num_obs:
        raise NotImplementedError()

    _temp = []
    rule_obs = [uniq_obs[x] for x in info["display_text"]["observations"]]
    sel_rules = rule_id_similarity_map[info["id"]][:n_least_sim_rules]

    if len(rule_obs) + 1 < num_obs:
        return []

    _obs = set()
    for r in sel_rules:
        r = rule_info[r]["display_text"]["observations"]
        _obs = _obs.union(set([uniq_obs[x] for x in r]))

    # print("Used Rules {} and Observations {}".format(len(sel_rules), len(_obs)))

    if number_random_sample < len(list(set(_obs) - set(rule_obs))):
        possible_choices = random.sample(
            list(set(_obs) - set(rule_obs)), number_random_sample
        )
    else:
        possible_choices = list(set(_obs) - set(rule_obs))

    obs_choice = random.sample(rule_obs, num_obs - 1)

    for r in possible_choices:
        _temp.append({"answer": r, "rule_choice": [r], "other_choice": obs_choice})

    cleaned = []
    for _t in _temp:
        all_options = _t["rule_choice"] + _t["other_choice"]
        ind = list(range(len(all_options)))
        random.shuffle(ind)
        ind = np.array(ind)
        answer_ind = ind == 0

        cleaned.append(
            {
                "answer_str": _t["answer"],
                "options": [all_options[x] for x in ind],
                "option_ids": [options_string[i] for i in range(len(all_options))],
                "question_type": "negative",
                "correct": answer_ind.tolist(),
            }
        )

    return cleaned


def create_dataset(
    rule_information,
    obs_mapping,
    asset_description: Dict[str, str],
    rule_id_similarity_map: Dict[str, List[str]],
    config: DatasetCreationConfig,
) -> dict[str, str | dict[str, Any]]:
    _temp = {}
    _temp["date"] = datetime.now().isoformat()
    _temp["questions"] = []
    _temp["unique_observations"] = obs_mapping
    _temp["asset_descriptions"] = asset_description
    _temp["creation_config"] = config.to_config()
    _temp["rule_id_similarity_map"] = rule_id_similarity_map

    for d in rule_information["rule_set"]:
        q = create_question(
            info=d,
            uniq_obs=obs_mapping,
            qid=len(_temp["questions"]),
            asset_description=asset_description,
            rule_id_similarity_map=rule_id_similarity_map,
            rule_info={k["id"]: k for k in rule_information["rule_set"]},
            config=config,
        )

        _temp["questions"].extend(q)

    postprocessed = []
    for q in _temp["questions"]:
        postprocessed.append(q.to_dict())

    _temp["rule_info"] = copy.deepcopy(rule_information)
    for d in _temp["rule_info"]["rule_set"]:
        d["display_text"]["rules"] = get_tree_repr(d["display_text"]["rules"])

    _temp["questions"] = postprocessed
    return _temp
