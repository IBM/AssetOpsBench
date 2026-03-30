import argparse
import json
import os
import time
from tenacity import retry, stop_after_attempt, wait_fixed

from dotenv import load_dotenv

load_dotenv()

from agent_hive.task import Task

from agent_hive.tools.skyspark import (
    iot_bms_tools,
    iot_bms_fewshots,
    iot_agent_description,
    iot_agent_name,
)
from agent_hive.tools.fmsr import (
    fmsr_tools,
    fmsr_fewshots,
    fmsr_task_examples,
    fmsr_agent_name,
    fmsr_agent_description,
)
from agent_hive.tools.tsfm import (
    tsfm_tools,
    tsfm_fewshots,
    tsfm_agent_name,
    tsfm_agent_description,
)
from agent_hive.tools.wo import (
    wo_agent_description,
    wo_agent_name,
    wo_fewshots,
    wo_tools,
)
from agent_hive.workflows.planning_review import PlanningReviewWorkflow

from agent_hive.workflows.sequential import SequentialWorkflow
from agent_hive.agents.react_reflect_agent import ReactReflectAgent
from agent_hive.agents.wo_agent import WorderOrderAgent
from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

import warnings

warnings.filterwarnings("ignore")

PLAN_PREFIX = os.path.dirname(os.path.abspath(__file__)) + "/plan/"


@retry(stop=stop_after_attempt(7), wait=wait_fixed(2))
def run_planning_workflow(question, llm_model, qid):
    iot_rr_agent = ReactReflectAgent(
        name=iot_agent_name,
        description=iot_agent_description,
        tools=iot_bms_tools,
        llm=llm_model,
        few_shots=iot_bms_fewshots,
        reflect_step=1
    )

    fmsr_rr_agent = ReactReflectAgent(
        name=fmsr_agent_name,
        description=fmsr_agent_description,
        tools=fmsr_tools,
        llm=llm_model,
        task_examples=fmsr_task_examples,
        few_shots=fmsr_fewshots,
        reflect_step=1
    )

    tsfm_rr_agent = ReactReflectAgent(
        name=tsfm_agent_name,
        description=tsfm_agent_description,
        tools=tsfm_tools,
        llm=llm_model,
        few_shots=tsfm_fewshots,
        reflect_step=1
    )
    
    wo_rr_agent = WorderOrderAgent(
        name=wo_agent_name,
        description=wo_agent_description,
        tools=wo_tools,
        llm=llm_model,
        few_shots=wo_fewshots,
        reflect_step=1
    )

    task_1 = Task(
        description=question,
        expected_output="",
        agents=[iot_rr_agent, fmsr_rr_agent, tsfm_rr_agent, wo_rr_agent],
    )

    wf = PlanningReviewWorkflow(
        tasks=[task_1],
        llm=llm_model
    )

    return wf.run(enable_summarization=True)



def run_react_reflect(utterance_file, react_llm_model_id, reverse=False):
    with open(utterance_file, "r") as json_file:
        data = json.load(json_file)

    if reverse:
        data = data[::-1]

    for utterance in data:
        print(
            f"ID: {utterance['id']}, Text: {utterance['text']}, model: {react_llm_model_id}, ReactReflectAgent..."
        )
        trajectory_file = f"trajectory/ReactReflectAgent/Model_{react_llm_model_id}_Q_{utterance['id']}_trajectory_output.json"

        if os.path.exists(trajectory_file):
            print(f"Skipping {utterance['id']}")
            continue

        start_time = time.time()
        ans = run_planning_workflow(
            utterance["text"],
            react_llm_model_id,
            utterance["id"],
        )

        end_time = time.time()
        runtime = end_time - start_time

        output = {
            "id": utterance["id"],
            "text": utterance["text"],
            "runtime": runtime,
            "trajectory": ans,
        }

        with open(trajectory_file, "w") as file:
            json.dump(output, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=int, default=15)
    parser.add_argument('-f', type=str)
    parser.add_argument("--reverse", type=bool, default=False)

    args = parser.parse_args()
    run_react_reflect(args.f, args.llm, reverse=args.reverse)
