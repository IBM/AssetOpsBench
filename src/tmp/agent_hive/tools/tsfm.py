from tsfmagent.agents.tsfmagent.tsfmfewshots import TSFM_FEW_SHOTS
from tsfmagent.agents.tsfmagent.tsfm_agent import getTSFMTools

tsfm_agent_name = 'Time Series Analytics and Forecasting'
tsfm_agent_description = ('Can assist with time series analysis, forecasting, anomaly '
                          'detection, and model selection, and supports pretrained models, context length '
                          'specifications, and regression tasks for various time series data')
tsfm_tools = getTSFMTools()
tsfm_fewshots = TSFM_FEW_SHOTS
