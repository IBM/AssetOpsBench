from meta_agent.default_meta_agent import AgentHub
import json
import os
from datetime import datetime
import time
import traceback
from reactxen.utils.model_inference import modelset
from meta_agent.agents.distractor_agents import load_prebuilt_agents

NUM_REFLECT = 1

def test_agent_hub_methods():
    # Create an instance of AgentHub
    agent_hub = AgentHub()

    # 1. Load default agents and tools
    print("Loading agents and tools...")
    agent_hub.load_default_agents()

    # for distractor agents
    # prebuilt_agents = load_prebuilt_agents()
    # for name, agent in prebuilt_agents.items():
    #     agent_hub.add_agent(name, agent, agent.get_examples())

    # 2. Check if specific agents and tools are loaded
    print("Checking loaded agents and tools...")
    print(f"Loaded agents: {list(agent_hub.agents.keys())}")
    assert "IoTAgent" in agent_hub.agents
    assert "TSFMAgent" in agent_hub.agents
    assert "JSONReader" in agent_hub.agents

    # 3. Show examples for a specific agent
    agent_hub.display_agents_and_examples()

    iotjsondata = open('src/meta_agent/scenarios/single_agent/iot_utterance_meta.json', 'r')
    fmsrjsondata = open('src/meta_agent/scenarios/single_agent/fmsr_utterance.jsonold', 'r')
    tsfmjsondata = open('src/meta_agent/scenarios/single_agent/tsfm_utterance.json', 'r')
    wojsondata = open('src/meta_agent/scenarios/single_agent/wo_utterance.json', 'r')
    end2endjsondata = open('src/meta_agent/scenarios/multi_agent/end2end_utterance.json')

    nowTime = datetime.now() 
    startRun = nowTime.replace(microsecond=0)
    startISO = startRun.isoformat()

    # "mistralai/mistral-large",  # 6
    # "meta-llama/llama-3-405b-instruct",  # 7
    # "meta-llama/llama-3-3-70b-instruct",  # 12
    # "meta-llama/llama-4-maverick-17b-128e-instruct-fp8", #16
    # "meta-llama/llama-4-scout-17b-16e-instruct", #17
    # "openai-azure/gpt-4.1-2025-04-14", #18
    # "ibm/granite-3-3-8b-instruct",  # 19

    for MODEL_ID in [6, 7, 12, 16, 17, 18, 19]:
        MODEL_SHORT_NAME = modelset[MODEL_ID]

        for file in [iotjsondata, fmsrjsondata, tsfmjsondata, wojsondata, end2endjsondata]:

            print(f'*** now loading {file}')
            file.seek(0) # rewind file
            utterances = json.load(file)

            for utterance in utterances:

                success = False
                for _ in range(5):
                    try:
                        process(agent_hub, startISO, MODEL_ID, MODEL_SHORT_NAME, utterance)
                        success = True
                    except BaseException as e:
                        print(f'EXCEPTION! {str(e)} ModelID: {MODEL_ID}, ModelName: {MODEL_SHORT_NAME}, utteranceID: {utterance["id"]}')
                        print(traceback.format_exc())

                    if success:
                        break

                    print('SLEEPING...')
                    time.sleep(60)
                
                if not success:
                    print(f'GIVING UP')
               

def process(agent_hub, startISO, MODEL_ID, MODEL_SHORT_NAME, utterance):

    utteranceID = utterance['id']

    print('\n******************************************')

    startTime = datetime.now()

    print(f"START-META: ISO {startISO} model = {MODEL_SHORT_NAME} ID: {utteranceID}, Text: {utterance['text']}, num_reflect_iter: {NUM_REFLECT}")
    agent = agent_hub.run(utterance['text'], model_id=MODEL_ID, outer_loop_step=NUM_REFLECT) # , agent_order_shift=4) 

    endTime = datetime.now()
    runTime = endTime - startTime
    runMinutes = runTime.total_seconds() / 60.0

    metric = agent.export_benchmark_metric()

    dir = "trajs/" +  startISO + '/' + MODEL_SHORT_NAME 

    os.makedirs(dir, exist_ok=True)

    fp = open(dir + '/' + str(utteranceID).zfill(4), 'w')
    traj = agent.export_trajectory()
    traj['overall_metric'] = metric # add to trajectory
    json.dump(traj, fp, indent=2)
    fp.close()

    # print(f'FINISH id={utterance["id"]}')
    print(f'FINISH-META: ISO {startISO} ModelID: {MODEL_ID}, ModelName: {MODEL_SHORT_NAME}, utteranceID: {utterance["id"]}, processing time {runMinutes} minutes, METRIC {json.dumps(metric)}')

# Execute the test script
test_agent_hub_methods()
