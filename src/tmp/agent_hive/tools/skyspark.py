from iotagent.demo.skysparkfewshots import SKYSPARK1
from iotagent.demo.run_reactreflect import getTools

iot_agent_name = 'IoT Data Download'
iot_agent_description = ('Can provide information about IoT sites, asset details, sensor data, and retrieve historical '
                         'data and metadata for various assets and equipment')
iot_tools = getTools()[0]
iot_fewshots = SKYSPARK1
