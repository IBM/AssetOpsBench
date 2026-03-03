from dotenv import load_dotenv

load_dotenv()

from reactxen.utils.model_inference import azure_openai_llm, watsonx_llm
import os
import json
from reactxen.agents.evaluation_agent.agent import EvaluationAgent
import re


ret = watsonx_llm(
    prompt="what is the capital of usa",
    model_id="openai-azure/gpt-4.1-2025-04-14",
)

print(ret)

# id range
fmsr_range = [101, 120]
iot_range = [1, 48]
tsfm_range = [201, 223]
wo_range = [400, 435]
multi_range = []
utterances = {}


def load_utterances():
    utterance_files = [
        "./scenarios/single_agent/iot_utterance_meta.json",
        "./scenarios/single_agent/fmsr_utterance.json",
        "./scenarios/single_agent/tsfm_utterance.json",
        "./scenarios/single_agent/wo_utterance.json",
        "./scenarios/multi_agent/end2end_utterance.json",
    ]
    for filepath in utterance_files:
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
                for d in data:
                    utterances[d["id"]] = d
            except json.JSONDecodeError as e:
                print(f"Error decoding {filename}: {e}")
    return utterances


def llm_eval(data, characteristic):
    eval_agent = EvaluationAgent(model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8")
    assert characteristic["text"] == data['text']
    print(characteristic["text"])
    print(data['text'])
    print('='*10)
    agent_think = 'The agent executes the following steps: '
    for item in data['trajectory']:
        agent_think += f"{item['task_number']}. task: {item['task_description']}; agent: {item['agent_name']}; response: {item['response']}. "
    final_answer = data['trajectory'][-1]['response']
    if final_answer.strip() == "":
        final_answer = data['trajectory'][-2]['response']
    try:
        review_resultFull = eval_agent.evaluate_response(
            question=characteristic["text"],
            agent_think=agent_think,
            agent_response=final_answer,
            characteristic_answer=characteristic["characteristic_form"],
        )
        print(review_resultFull)
        
        return review_resultFull
    except BaseException as e:
        print(f"EXCEPTION: {e}")
    
    return None


utterances = load_utterances()
print(utterances)

# Directory path
directory = (
    "./src/agent_hive/tests/multi_agent/trajectory/ReactReflectAgent"
)

# Store all loaded data
all_logs = {}

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
                all_logs[filename] = data
                # eval
                if 'mvk_evaluation' not in data:
                    evaluation_res = llm_eval(data, utterances[int(re.findall(r'\d+', filename)[1])])
                    if evaluation_res:
                        data["mvk_evaluation"] = evaluation_res
                    
                    with open(filepath, "w") as f:
                        json.dump(data, f, indent=4)
            except json.JSONDecodeError as e:
                print(f"Error decoding {filename}: {e}")

print(f"Loaded {len(all_logs)} JSON files.")
