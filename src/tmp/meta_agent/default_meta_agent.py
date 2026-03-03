from meta_agent.meta_agent import MetaAgent
from meta_agent.agents.pre_built_agents import load_prebuilt_agents
from meta_agent.tools.pre_built_tools import load_prebuilt_tools
from meta_agent.agents.IoT.IoTAgentFewShots import get_iot_agent_examples
from meta_agent.agents.FMSR.FMSRAgentFewShots import get_fmsr_agent_examples
from meta_agent.agents.TSFM.TSFMAgentFewShots import get_tsfm_agent_examples
from meta_agent.agents.RuleLogic.RuleLogicAgentFewShots import (
    get_rulelogic_agent_examples,
)
from meta_agent.agents.WorkOrder.WorkOrderFewShots import get_workorder_agent_examples


class AgentHub(MetaAgent):
    def load_default_agents(self) -> None:
        """Load prebuilt agents and tools into the system."""

        # Load prebuilt agents and tools using functions
        prebuilt_agents = load_prebuilt_agents()
        for name, agent in prebuilt_agents.items():
            self.add_agent(name, agent)

        # Register examples dynamically from example functions
        self.add_agent(
            "IoTAgent", 
            prebuilt_agents["IoTAgent"], 
            examples=get_iot_agent_examples()
        )
        self.add_agent(
            "FMSRAgent",
            prebuilt_agents["FMSRAgent"],
            examples=get_fmsr_agent_examples(),
        )
        self.add_agent(
            "TSFMAgent",
            prebuilt_agents["TSFMAgent"],
            examples=get_tsfm_agent_examples(),
        )
        self.add_agent(
            "RuleLogicAgent",
            prebuilt_agents["RuleLogicAgent"],
            examples=get_rulelogic_agent_examples(),
        )
        self.add_agent(
            "WorkOrderAgent",
            prebuilt_agents["WorkOrderAgent"],
            examples=get_workorder_agent_examples(),
        )

        # Load prebuilt tools
        prebuilt_tools = load_prebuilt_tools()
        for name, tool in prebuilt_tools.items():
            self.add_agent(name, tool)

        print("All prebuilt agents and tools have been successfully loaded.")
