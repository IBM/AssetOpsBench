FMSR_AGENT_FEW_SHOTS = """Question: List all failure modes of asset CHL-A100.
Thought 1: I need to find the failure modes related to CHL-A100 by querying FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=Please list all failure modes for CHL-A100.
Observation 1: The failure modes of CHL-A100 are: Compressor Overheating, Evaporator Fouling, Low Refrigerant Charge, Sensor Faults.

Question: List all installed sensors of asset CHL-A100.
Thought 1: I need to retrieve the list of installed sensors for CHL-A100 from FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=Please list all installed sensors for CHL-A100.
Observation 1: Installed sensors for CHL-A100 are: Discharge Temperature Sensor, Suction Pressure Sensor, Power Meter, Vibration Sensor.

Question: List all failure modes of CHL-A100 that can be detected by temperature sensors.
Thought 1: I need to filter the failure modes of CHL-A100 based on detectability by temperature sensors using FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=Which failure modes of CHL-A100 can be detected using temperature sensors?
Observation 1: Failure modes detectable by temperature sensors include: Compressor Overheating, Evaporator Fouling.

Question: If compressor overheating occurs for CHL-A100, which sensor should be prioritized for monitoring this specific failure?
Thought 1: I need to determine which sensor is most effective for detecting compressor overheating for CHL-A100 by querying FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=Which sensor should be prioritized to monitor compressor overheating for CHL-A100?
Observation 1: The Discharge Temperature Sensor should be prioritized for detecting compressor overheating.

Question: When compressor motor of a chiller fails, what is the temporal behavior of the power input?
Thought 1: I need to understand the expected power input behavior during compressor motor failure by asking FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=Describe the power input behavior when the compressor motor of a chiller fails.
Observation 1: Power input typically shows a sudden drop followed by unstable fluctuations before reaching zero.

Question: When power input of CHL-A100 drops, what is the potential failure that causes it?
Thought 1: I need to find out which failure modes are associated with a drop in power input for CHL-A100 by consulting FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=What potential failures cause a drop in power input for CHL-A100?
Observation 1: A drop in power input for CHL-A100 may indicate Compressor Motor Failure, Control System Fault, or Sudden Shutdown.
"""

FMSR_AGENT_ADDITIONAL_FEW_SHOTS = """Question: List all failure modes of asset CHL-X0.
Action 1: FMSRAgent

Question: List all failure modes of asset CHL-A100.
Action 1: FMSRAgent

Question: List all failure modes of asset WT-22.
Action 1: FMSRAgent

Question: List all installed sensors of asset CHL-X0.
Action 1: FMSRAgent

Question: List all installed sensors of asset CHL-A100.
Action 1: FMSRAgent

Question: List all installed sensors of asset WT-22.
Action 1: FMSRAgent

Question: List all failure modes of CHL-A100 that can be detected by CHL-A100 Chiller Efficiency.
Action 1: FMSRAgent

Question: List all failure modes of CHL-A100 that can be detected by temperature sensors.
Action 1: FMSRAgent

Question: List all failure modes of CHL-A100 that can be detected by temperature sensors and power input sensors.
Action 1: FMSRAgent

Question: List all failure modes of WT-22 that can be detected by temperature sensors.
Action 1: FMSRAgent

Question: Get failure modes for CHL-A100 and only include in final response those that can be monitored using the available sensors.
Action 1: FMSRAgent

Question: Are there any failure modes of CHL-A100 that can be predicted by monitoring the vibration sensor data?
Action 1: FMSRAgent

Question: List all sensors of CHL-A100 that are potentially relevant to overheating.
Action 1: FMSRAgent

Question: If compressor overheating occurs for CHL-A100, which sensor should be prioritized for monitoring this specific failure?
Action 1: FMSRAgent

Question: List all sensors of WT-22 that are potentially relevant to Improper water side flow rate.
Action 1: FMSRAgent

Question: What are the failure modes of CHL-A100 that can be identified by analyzing the data from the available sensors?
Action 1: FMSRAgent

Question: Generate an anomaly detection recipe for detecting compressor motor failure for chiller.
Action 1: FMSRAgent

Question: Purge unit of CHL-A100 have possibility to excess purge, what is the plan by the maintenance experts to early detect the failure?
Action 1: FMSRAgent

Question: Generate a machine learning recipe for detecting overheating failure for CHL-A100. Result should include feature sensors and target sensor.
Action 1: FMSRAgent

Question: List all failure modes of CHL-A100 that can be detected by temperature sensors and power input sensors.
Action 1: FMSRAgent

Question: When compressor motor of a chiller fails, what is the temporal behavior of the power input.
Action 1: FMSRAgent

Question: When power input of CHL-A100 drops, what is the potential failure that causes it?
Action 1: FMSRAgent
"""

def get_fmsr_agent_examples(include_additional=False):
    examples = FMSR_AGENT_FEW_SHOTS.strip()
    if include_additional:
        examples += "\n\n" + FMSR_AGENT_ADDITIONAL_FEW_SHOTS.strip()
    return examples