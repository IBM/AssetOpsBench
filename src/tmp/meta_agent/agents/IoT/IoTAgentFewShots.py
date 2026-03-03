IOT_AGENT_FEW_SHOTS = """Question: send me the values for CU02004 at SiteX for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP"
Thought 1: I need to download asset history for CU02004 at SiteX from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00 for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP".
Action 1: IoTAgent
Action Input 1: request=download asset history for CU02004 at SiteX from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00 for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP"
Observation 1: {"site_name": "SiteX", "assetnum": "CU02004", "total_observations": 25, "start": "2025-03-26T00:00:00.000000+00:00", "final": "2025-04-02T00:00:00.000000+00:00", "file_path": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/c328516a-643f-40e6-8701-e875b1985c38.json", "message": "found 25 observations. file_path contains a JSON array of Observation data"}

Question: what sites are there
Action 1: IoTAgent
Action Input 1: request=list all the sites
Observation 1: {"sites": ["SiteX", "SiteY", "SiteZ"], "message": "List of available sites retrieved."}

Question: what assets are at site SiteX
Action 1: IoTAgent
Action Input 1: request=list assets for site SiteX
Observation 1: {"site_name": "SiteX", "assets": ["CU02004", "CU02005", "CU02006"], "message": "List of assets at SiteX retrieved."}

Question: download sensor data for CU02004 at SiteX
Action 1: IoTAgent
Action Input 1: request=download sensor data for CU02004 at SiteX
Observation 1: {"site_name": "SiteX", "assetnum": "CU02004", "sensor_data": "data_path=/path/to/sensor_data.json", "message": "Sensor data for CU02004 downloaded."}

Question: download asset history for CU02004 at SiteX from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00 for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP"
Action 1: IoTAgent
Action Input 1: request=download asset history for CU02004 at SiteX from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00 for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP"
Observation 1: {"site_name": "SiteX", "assetnum": "CU02004", "total_observations": 25, "start": "2025-03-26T00:00:00.000000+00:00", "final": "2025-04-02T00:00:00.000000+00:00", "file_path": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/c328516a-643f-40e6-8701-e875b1985c38.json", "message": "found 25 observations. file_path contains a JSON array of Observation data"}

Question: merge these JSON files file1.json and file2.json into a single JSON file
Action 1: IoTAgent
Action Input 1: request=merge JSON files file1.json and file2.json
Observation 1: {"file1": "/path/to/file1.json", "file2": "/path/to/file2.json", "merged_file": "/path/to/merged_file.json", "message": "Files merged successfully."}

Question: How do I get a list of properties from a JSON file
Action 1: IoTAgent
Action Input 1: request=list properties from JSON file
Observation 1: {"json_file": "/path/to/file.json", "properties": ["property1", "property2", "property3"], "message": "List of properties extracted from JSON file."}

Question: I need to read the JSON file 0001.json
Action 1: IoTAgent
Action Input 1: request=read JSON file 0001.json
Observation 1: {"file_path": "/path/to/0001.json", "content": {"key1": "value1", "key2": "value2"}, "message": "JSON file 0001.json read successfully."}

Question: how do I calculate the start date for last week or past week?
Action 1: IoTAgent
Action Input 1: request=calculate start date for last week
Observation 1: {"last_week_start_date": "2025-04-10T00:00:00", "message": "Start date for last week calculated successfully."}
"""

"""
When passing information from one tool to another, the preferred method is to pass the filename returned by the first tool to the second tool. when passing filenames to a tool, pass the full absolute path.
"""

IOT_AGENT_ADDITIONAL_FEW_SHOTS = """
Question: Check if the sensor values in /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/809b66b8-3e13-483a-95eb-dcce5b29e690.json fall within the allowable range.
Thought 1: I need to read the file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/809b66b8-3e13-483a-95eb-dcce5b29e690.json and verify whether the values for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP" are within their allowable ranges.
Action 1: IoTAgent
Action Input 1: request=read and validate sensor values from JSON file at /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/809b66b8-3e13-483a-95eb-dcce5b29e690.json
Observation 1: {"chilled_water_leaving_temp_in_range": true, "chilled_water_return_temp_in_range": false, "message": "Validation completed. Successfully checked values against allowable ranges."}
"""

def get_iot_agent_examples(include_additional=False):
    examples = IOT_AGENT_FEW_SHOTS.strip()
    if include_additional:
        examples += "\n\n" + IOT_AGENT_ADDITIONAL_FEW_SHOTS.strip()
    return examples