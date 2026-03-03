TSFMAGENTFEWSHOTS = """Question: Analyze the time-series data in file xxxxx.json and look for anomalies in Chiller 6 Return Temperature.
Thought 1: I need to perform time-series feature monitoring on Chiller 6 Return Temperature.
Action 1: TSFMAgent
Action Input 1: request=Analyze the time-series data in file xxxxx.json and detect anomalies in Chiller 6 Return Temperature.
Observation 1: Completed anomaly detection analysis for Chiller 6 Return Temperature.

Question: Analyze file chiller_power_data.json and identify any sudden spikes in Chiller 7 power consumption.
Thought 1: I need to analyze the time-series data for sudden spikes in power consumption.
Action 1: TSFMAgent
Action Input 1: request=Analyze file chiller_power_data.json and detect sudden spikes in Chiller 7 power consumption.
Observation 1: Completed spike detection analysis for Chiller 7 power consumption.

Question: Forecast the next 24 hours of Evaporator Leaving Water Temperature for CHL-X0 based on file evap_temp_data.json.
Thought 1: I need to build a forecast model using the time-series data.
Action 1: TSFMAgent
Action Input 1: request=Forecast next 24 hours of Evaporator Leaving Water Temperature using evap_temp_data.json for CHL-X0.
Observation 1: Completed 24-hour forecast for Evaporator Leaving Water Temperature for CHL-X0.
"""

def get_tsfm_agent_examples():
    return TSFMAGENTFEWSHOTS.strip()
