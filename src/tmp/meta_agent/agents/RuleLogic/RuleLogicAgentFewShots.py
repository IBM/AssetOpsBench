RULELOGICAGENTFEWSHOTS = """Question: I'd be interested in setting up an alert if Chiller 6 is operational, the difference between supply temperature and liquid refrigerant evaporator temperature exceeds 4 degrees Fahrenheit, and this condition persists for at least three hours.
Thought 1: I need to generate a rule based on sensor data conditions.
Action 1: RuleLogicAgent
Action Input 1: Set up an alert for Chiller 6 when operational, the difference between supply temperature and liquid refrigerant evaporator temperature exceeds 4°F for at least three hours.
Observation 1: Generated rule for Chiller 6 as specified.

Question: Analyze the time-series data in file xxxxx.json and look for anomalies in Chiller 6 Return Temperature.
Thought 1: I need to perform anomaly detection on Chiller 6 Return Temperature.
Action 1: RuleLogicAgent
Action Input 1: Analyze time-series data in file xxxxx.json for anomalies in Chiller 6 Return Temperature.
Observation 1: Completed anomaly detection analysis for Chiller 6 Return Temperature.

Question: Set an alert if Chiller 5 condenser pressure exceeds 250 psi for more than 10 minutes.
Thought 1: I need to create a rule to monitor Chiller 5 condenser pressure.
Action 1: RuleLogicAgent
Action Input 1: Set an alert if Chiller 5 condenser pressure exceeds 250 psi for more than 10 minutes.
Observation 1: Created rule to monitor Chiller 5 condenser pressure.

Question: Generate a rule to trigger an alert if the cooling tower fan motor current drops below 3 Amps while the fan is running.
Thought 1: I need to create a rule based on fan motor current and operational status.
Action 1: RuleLogicAgent
Action Input 1: Create a rule for cooling tower fan motor to trigger an alert if running and motor current drops below 3 Amps.
Observation 1: Rule generated for cooling tower fan motor monitoring.

Question: Perform an analysis on file cooling_tower_data.json and identify if there are any abnormal trends in fan motor current.
Thought 1: I need to detect anomalies in fan motor current data.
Action 1: RuleLogicAgent
Action Input 1: Analyze cooling_tower_data.json for abnormal trends in fan motor current.
Observation 1: Completed abnormal trend detection for fan motor current.
"""

def get_rulelogic_agent_examples():
    return RULELOGICAGENTFEWSHOTS.strip()
