from meta_agent.agents.IoT.IoTWrapper import IoTAgentFunctions
from meta_agent.agents.TSFM.TSFMWrapper import TSFMAgentFunctions
from meta_agent.agents.TSFM.TSFMTool import TSFMAgent
from meta_agent.agents.FMSR.FMSRWrapper import FMSRAgentFunctions
from meta_agent.agents.FMSR.FMSRTool import FMSRAgent
from meta_agent.agents.IoT.IoTTool import IoTAgent
from meta_agent.agents.WorkOrder.WorkOrderTool import WOAgent
from meta_agent.agents.WorkOrder.WorkOrderWrapper import WOAgentFunctions
from meta_agent.agents.RuleLogic.RuleLogicTool import RuleLogicAgent
from meta_agent.agents.RuleLogic.RuleLogicWrapper import RuleLogicAgentFunctions
from typing import Dict

# Agent creation functions
def create_iot_agent() -> IoTAgent:
    """Create and return an IoTAgent instance."""
    functions = IoTAgentFunctions()
    return IoTAgent(iotAgentFunctions=functions)


def create_tsfm_agent() -> TSFMAgent:
    """Create and return a TSFMAgent instance."""
    fns = TSFMAgentFunctions()
    return TSFMAgent(tsfmAgentFunctions=fns)


def create_fmsr_agent() -> FMSRAgent:
    """Create and return an FMSRAgent instance."""
    fmsr_functions = FMSRAgentFunctions()
    return FMSRAgent(fmsrAgentFunctions=fmsr_functions)

def create_wo_agent() -> WOAgent:
    """Create and return a WOAgent instance."""
    wo_functions = WOAgentFunctions()
    return WOAgent(tsfmAgentFunctions=wo_functions)

def create_rulelogic_agent() -> RuleLogicAgent:
    """Create and return a RuleLogic Agent instance."""
    wo_functions = RuleLogicAgentFunctions()
    return RuleLogicAgent(rulelogicAgentFunctions=wo_functions)

# Centralized function to load all prebuilt agents
def load_prebuilt_agents() -> Dict[str, object]:
    """
    Load prebuilt agents and return them as a dictionary.
    
    The dictionary keys are agent names, and values are the corresponding agent instances.
    """
    agents = {
        "IoTAgent": create_iot_agent(),
        "TSFMAgent": create_tsfm_agent(),
        "FMSRAgent": create_fmsr_agent(),
        "WorkOrderAgent": create_wo_agent(),
        "RuleLogicAgent": create_rulelogic_agent(),
    }
    return agents
