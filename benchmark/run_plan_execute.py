import argparse
import json
import os
from dotenv import load_dotenv

load_dotenv()

from agent_hive.task import Task
from agent_hive.tools.fmsr import (
    fmsr_tools,
    fmsr_fewshots,
    fmsr_task_examples,
    fmsr_agent_name,
    fmsr_agent_description,
)
from agent_hive.tools.skyspark import (
    iot_bms_tools,
    iot_bms_fewshots,
    iot_agent_description,
    iot_agent_name,
)

from agent_hive.workflows.planning import PlanningWorkflow
from agent_hive.workflows.planning_review import PlanningReviewWorkflow
from agent_hive.agents.react_agent import ReactAgent
from agent_hive.agents.react_reflect_agent import ReactReflectAgent

from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

import warnings

warnings.filterwarnings("ignore")

PLAN_PREFIX = os.path.dirname(os.path.abspath(__file__)) + "/plan/"

UTTERRANCE_FILE = '/home/scenarios/multi_agent/end2end_utterance.json'


def run_planning_workflow_with_reflection(
        question, llm_model, qid, workflow, generate_steps_only=False
):
    iot_rr_agent = ReactReflectAgent(
        name=iot_agent_name,
        description=iot_agent_description,
        tools=iot_bms_tools,
        llm=llm_model,
        few_shots=iot_bms_fewshots,
    )

    fmsr_rr_agent = ReactReflectAgent(
        name=fmsr_agent_name,
        description=fmsr_agent_description,
        tools=fmsr_tools,
        llm=llm_model,
        task_examples=fmsr_task_examples,
        few_shots=fmsr_fewshots,
    )

    task_1 = Task(
        description=question,
        expected_output="",
        agents=[iot_rr_agent, fmsr_rr_agent],
    )

    wf = PlanningWorkflow(
        tasks=[task_1],
        llm=llm_model,
    )
    if workflow == "pr":
        wf = PlanningReviewWorkflow(
            tasks=[task_1],
            llm=llm_model,
        )

    if generate_steps_only:
        return wf.generate_steps(
            save_plan=True,
            saved_plan_filename=PLAN_PREFIX + f"Model_{llm_model}_Q_{qid}_plan",
        )

    return wf.run(enable_summarization=True)


def run_planning_workflow(
        question, llm_model, qid, workflow, generate_steps_only=False
):
    iot_r_agent = ReactAgent(
        name=iot_agent_name,
        description=iot_agent_description,
        tools=iot_bms_tools,
        llm=llm_model,
        few_shots=iot_bms_fewshots,
    )

    fmsr_r_agent = ReactAgent(
        name=fmsr_agent_name,
        description=fmsr_agent_description,
        tools=fmsr_tools,
        llm=llm_model,
        task_examples=fmsr_task_examples,
        few_shots=fmsr_fewshots,
    )

    task_2 = Task(
        description=question,
        expected_output="",
        agents=[iot_r_agent, fmsr_r_agent],
    )

    wf = PlanningWorkflow(
        tasks=[task_2],
        llm=llm_model,
    )
    if workflow == "pr":
        wf = PlanningReviewWorkflow(
            tasks=[task_2],
            llm=llm_model,
        )

    if generate_steps_only:
        return wf.generate_steps(
            save_plan=True,
            saved_plan_filename=PLAN_PREFIX + f"Model_{llm_model}_Q_{qid}_plan",
        )

    return wf.run(enable_summarization=True)


def run_react(utterance_file, react_llm_model_id, workflow="pr", generate_steps_only=False, reverse=False):
    # Load the JSON data from the file
    with open(utterance_file, "r") as json_file:
        data = json.load(json_file)

    if reverse:
        data = data[::-1]

    folder = "./trajectory/ReactAgent"
    os.makedirs(folder, exist_ok=True)

    for utterance in data:
        print(
            f"ID: {utterance['id']}, Text: {utterance['text']}, model: {react_llm_model_id}, ReactAgent..."
        )
        trajectory_file = f"trajectory/ReactAgent/Model_{react_llm_model_id}_Q_{utterance['id']}_trajectory_output.json"

        if os.path.exists(trajectory_file):
            print(f"Skipping {utterance['id']}")
            continue

        ans = run_planning_workflow(
            utterance["text"],
            react_llm_model_id,
            utterance["id"],
            workflow=workflow,
            generate_steps_only=generate_steps_only,
        )

        if generate_steps_only:
            continue

        output = {"id": utterance["id"], "text": utterance["text"], "trajectory": ans}

        with open(trajectory_file, "w") as file:
            json.dump(output, file, indent=4)


def run_react_reflect(utterance_file, react_llm_model_id, workflow="pr", generate_steps_only=False, reverse=False):
    # Load the JSON data from the file
    with open(utterance_file, "r") as json_file:
        data = json.load(json_file)

    if reverse:
        data = data[::-1]

    folder = "./trajectory/ReactReflectAgent"
    os.makedirs(folder, exist_ok=True)

    for utterance in data:
        print(
            f"ID: {utterance['id']}, Text: {utterance['text']}, model: {react_llm_model_id}, ReactReflectAgent..."
        )
        trajectory_file = f"trajectory/ReactReflectAgent/Model_{react_llm_model_id}_Q_{utterance['id']}_trajectory_output.json"

        if os.path.exists(trajectory_file):
            print(f"Skipping {utterance['id']}")
            continue

        ans = run_planning_workflow_with_reflection(
            utterance["text"],
            react_llm_model_id,
            utterance["id"],
            workflow=workflow,
            generate_steps_only=generate_steps_only,
        )

        if generate_steps_only:
            continue

        output = {"id": utterance["id"], "text": utterance["text"], "trajectory": ans}

        with open(trajectory_file, "w") as file:
            json.dump(output, file, indent=4)


if __name__ == "__main__":
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     executor.map(run, [15, 6, 10])
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance_file", type=str, default=UTTERRANCE_FILE)
    parser.add_argument("--llm", type=int, default=15)
    parser.add_argument(
        "--agent", type=str, choices=["react", "reflect"], default="react"
    )
    parser.add_argument("--workflow", type=str, choices=["p", "pr"], default="pr")
    parser.add_argument("--generate_steps_only", type=bool, default=False)
    parser.add_argument("--reverse", type=bool, default=False)

    args = parser.parse_args()

    if args.agent == "react":
        run_react(
            args.utterance_file,
            args.llm,
            workflow=args.workflow,
            generate_steps_only=args.generate_steps_only,
            reverse=args.reverse
        )
    elif args.agent == "reflect":
        run_react_reflect(
            args.utterance_file,
            args.llm,
            workflow=args.workflow,
            generate_steps_only=args.generate_steps_only,
            reverse=args.reverse
        )
    else:
        raise ValueError("Invalid agent choice")
