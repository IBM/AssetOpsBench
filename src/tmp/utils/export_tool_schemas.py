import json
import os
from langchain_core.utils.function_calling import convert_to_openai_function

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
    iot_task_examples,
)
from agent_hive.tools.tsfm import (
    tsfm_tools,
    tsfm_fewshots,
    tsfm_agent_name,
    tsfm_agent_description,
    tsfm_task_examples,
)
from agent_hive.tools.wo import (
    wo_agent_description,
    wo_agent_name,
    wo_fewshots,
    wo_tools,
    wo_task_examples,
)

# ----------------------------------------------------------
# Helper function to convert tools from a given agent module
# ----------------------------------------------------------
def convert_tools(tools, agent_name, agent_description, fewshots, examples):
    tool_schemas = []
    for tool in tools:
        schema = convert_to_openai_function(tool)
        schema["Agent"] = agent_name
        schema["Agent_Description"] = agent_description
        schema["Sample_Examples"] = examples
        schema["Fewshots"] = fewshots
        tool_schemas.append(schema)
    return tool_schemas


# ----------------------------------------------------------
# Collect tool schemas for all agents
# ----------------------------------------------------------
tSet = []

tSet.extend(convert_tools(fmsr_tools, fmsr_agent_name, fmsr_agent_description, fmsr_fewshots, fmsr_task_examples))
tSet.extend(convert_tools(iot_bms_tools, iot_agent_name, iot_agent_description, iot_bms_fewshots, iot_task_examples))
tSet.extend(convert_tools(tsfm_tools, tsfm_agent_name, tsfm_agent_description, tsfm_fewshots, tsfm_task_examples))
tSet.extend(convert_tools(wo_tools, wo_agent_name, wo_agent_description, wo_fewshots, wo_task_examples))

# ----------------------------------------------------------
# Save the collected schema specification to JSON
# ----------------------------------------------------------
output_path = os.path.join(os.getcwd(), "agent_tool_schemas.json")

with open(output_path, "w") as f:
    json.dump(tSet, f, indent=2)

print(f"✅ Exported {len(tSet)} tool schemas to {output_path}")
