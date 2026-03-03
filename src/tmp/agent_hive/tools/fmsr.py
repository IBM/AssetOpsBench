from reactxen.tools.jsonreader.jsonreader import JSONReader
from reactxen.tools.jsonreader.jsonwrapper import JSONWrapperFunctions

from fmsr_agent.agent.react import get_fmsr_task_examples, get_fmsr_fewshots
from fmsr_agent.tools import (
    GetFailureModeTool,
    GetFailureModeAndSensorMappingTool,
    GetSensorMetadataTool,
)

fns = JSONWrapperFunctions()
jsonReader = JSONReader(functions=fns)

# fmsr_tools = [GetFailureModeTool(), GetFailureModeAndSensorMappingTool(), GetSensorMetadataTool(), jsonReader]

fmsr_agent_name = "Failure Mode and Sensor Relevancy Expert for Industrial Asset"
fmsr_agent_description = (
    "Can provide information about failure modes, mapping between failure modes and sensors, "
    "and can generate machine learning recipes for specific failures"
)
fmsr_tools = [
    GetFailureModeTool(),
    GetFailureModeAndSensorMappingTool(),
    GetSensorMetadataTool(),
]
fmsr_fewshots = get_fmsr_fewshots()
fmsr_task_examples = get_fmsr_task_examples()
