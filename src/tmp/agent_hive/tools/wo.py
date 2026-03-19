from woagent.agents.woagent.wofewshots import WO_FEW_SHOTS
from woagent.agents.woagent.wo_agent import getWOTools

wo_agent_name = "Work Order Management"
wo_agent_description = (
    "Can retrieve, analyze, and generate work orders for equipment based on historical data, "
    "anomalies, alerts, and performance metrics, offering recommendations for preventive and "
    "corrective actions, including bundling, prioritization, and predictive maintenance"
)
wo_tools = getWOTools()
wo_fewshots = WO_FEW_SHOTS
