BMSFEWSHOTS = """Question: what sites are there
Thought 1: I need to get the list of available sites to answer the question.
Action 1: sites
Action Input 1: {}
Observation 1: {"sites": "MAIN"}
Thought 2: I now have the list of available sites
Action 2: Finish
Action Input 2: The sites are: "MAIN"

Question: what assets are at site MAIN
Thought 1: I need to get the list of assets at site MAIN to answer the question.
Action 1: assets
Action Input 1: site_name=MAIN
Observation 1: Available assets: found the following 483 assets, listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/9e17f07b-0b4a-43af-85c1-562dc21f105e.
Thought 2: I now have the list of assets for site MAIN, which is stored in a file.
Action 2: Finish
Action Input 2: The assets at site MAIN are listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/9e17f07b-0b4a-43af-85c1-562dc21f105e.

Question: download sensor data for Chiller 4 at MAIN site
Thought 1: I need to get the sensor data for Chiller 4 at site MAIN to answer the question.
Action 1: sensors
Action Input 1: assetnum=Chiller 4, site_name=MAIN
Observation 1: {"site_name": "MAIN", "assetnum": "Chiller 4", "total_sensors": 2, "file_path": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/c6571941-4857-4701-bd8a-9a28fa2435c3.json", "message": "found 2 sensors for assetnum Chiller 4 and site_name MAIN. file_path contains a JSON array of Sensor data"}
Thought 2: I now have the sensor data for Chiller 4 at site MAIN, which is stored in a file.
Action 2: Finish
Action Input 2: The sensor data for Chiller 4 at site MAIN has been downloaded and is listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/c6571941-4857-4701-bd8a-9a28fa2435c3.json.

Question: download asset history for Chiller 4 at MAIN site from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00
Thought 1: I need to get the history for Chiller 4 at MAIN site within the specified time range.
Action 1: history
Action input 1: site_name=MAIN, assetnum=Chiller 4, start=2025-03-26T00:00:00.000000+00:00, final=2025-04-02T00:00:00.000000+00:00
Observation 1: {"site_name": "MAIN", "assetnum": "Chiller 4", "total_observations": 3, "start": "2025-03-26T00:00:00.000000+00:00", "final": "2025-04-02T00:00:00.000000+00:00", "file_path": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/44934c53-0ad8-4551-adbe-5b11ad03865a.json", "message": "found 3 observations. file_path contains a JSON array of Observation data"}
Thought 2: I now have the history for asset Chiller 4 at MAIN site within the specified time range.
Action 2: Finish
Action Input 2: The asset history for Chiller 4 at MAIN site from 2025-03-26T00:00:00.000000+00:00 to 2025-04-02T00:00:00.000000+00:00 has been downloaded and is listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/44934c53-0ad8-4551-adbe-5b11ad03865a.json.

Question: merge these JSON files file1.json and file2.json into a single JSON file
Thought 1: I need to combine the data from JSON files file1.json and file2.json into a single file.
Action 1: jsonfilemerge
Action input 1: file_name_1=file1.json, file_name_2=file2.json
Observation 1: The files have been merged: Files file1.json and file2.json have been merged into file3.json.
Thought 2: I now have the merged data from file1.json and file2.json in a single file.
Action 2: Finish
Action Input 2: The contents of file1.json and file2.json have been successfully merged into file3.json.

Question: How do I get a list of properties from a JSON file
Thought 1: I need to call jsonproperties on the file
Action 1: jsonproperties
Action input 1: file_name_1=file3.json
Observation 1: { "properties": [ "a", "b", "c"] }
Thought 2: I now have the list of properties from the JSON file
Action 2: Finish
Action Input 2: the properties are a, b, c

Question: I need to read the JSON file 0001.json.
Thought 1: I need to read the contents of the JSON file to access its data.
Action 1: jsonreader
Action Input 1: file_name=0001.json
Observation 1: The contents of the JSON file have been read.
Thought 2: I now have the data from the JSON file.
Action 2: Finish
Action Input 2: The contents of 0001.json have been successfully read.

Question: how do I calculate the start date for last week or past week?
Thought 1: I should first obtain the current date and then subtract 7 days from the current date.
Action 1: currentdatetime
Action Input 1: {}
Observation 1: {"currentDateTime": "2025-01-08T19:40:04.664816+00:00", "currentDateTimeDescription": "Today's date is 2025-01-08 and time is 19:40:04."}
Thought 2: Now, I have the current date and I should proceed to generate the start date for last week.
Action 2: Self-Ask
Action Input 2: what is the start date for last week if current date is 2025-01-08T19:40:04.664816+00:00
Observation 2: based on current date, the last week starts on 2025-01-01T00:00:00.000000+00:00
Thought 3: Now I have the start date for last week
Action 3: Finish
Action Input 3: the start date for last week is 2025-01-01T00:00:00.000000+00:00.

"""

temp = """

Question: download asset history for sensor c5_B027_CH4_B/027_CH1_AVGERAGE_AMPS_() on Chiller 4 at MAIN site from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00
Thought 1: I need to get the history for sensor c5_B027_CH4_B/027_CH1_AVGERAGE_AMPS_() on Chiller 4 at MAIN site within the specified time range.
Action 1: history
Action input 1: site_name=MAIN, assetnum=Chiller 4, sensor_name_list=c5_B027_CH4_B/027_CH1_AVGERAGE_AMPS_(), start=2025-01-01T00:00:00.000000+00:00, final=2025-01-08T00:00:00.000000+00:00 
Observation 1: {"site_name": "MAIN", "assetnum": "Chiller 4", "total_observations": 0, "start": "2025-01-01T00:00:00.000000+00:00", "final": "2025-01-08T00:00:00.000000+00:00", "file_path": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/97fbb3de-f442-438a-88c4-989abfde2c21.json", "message": "found 0 observations. file_path contains a JSON array of Observation data"}
Thought 2: I now have the history for asset Chiller 4 at MAIN site within the specified time range.
Action 2: Finish
Action Input 2: The asset history for Chiller 4 at MAIN site from 2025-01-01T00:00:00.000000+00:00 to 2025-01-08T00:00:00.000000+00:00 has been downloaded and is listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/97fbb3de-f442-438a-88c4-989abfde2c21.json.

Question: how do I learn the correct name for the sensor Condenser Return Temperature on Chiller 1 at site MAIN?
Thought 1: I should first list all the sensors on the asset
Action 1: sensors
Action Input 1: asssetnum=Chiller 1, site_name=MAIN
Observation 1: {"site_name": "MAIN", "asssetnum": "Chiller 1", "total_sensors": 12, "file_path": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/383594ff-bdf4-4d1e-8947-f1f95ef0761c.json", "message": "found 12 sensors for asssetnum Chiller 1 and site_name MAIN. file_path contains a JSON array of Sensor data"}
Thought 2: I now have the list of sensors for Chiller 1 at site MAIN, which is stored in a file.
Action 2: jsonreader
Action Input 2: file_name=/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/383594ff-bdf4-4d1e-8947-f1f95ef0761c.json
Observation 2: [{"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Condenser Water Return To Tower Temperature"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Chiller Efficiency"}]
Thought 3: I should look at the sensor_name and pick the closest match to Condenser Return Temperature. I should use exactly the sensor name that is specified by the sensor_name property.
Action 3: Finish
Action Input 3: the correct sensor_name is Chiller 1 Condenser Water Return To Tower Temperature

"""

DHAVAL_EXTRA = """
Question: How assets are represented? Give five examples.
Thought 1: I need to get the list of assets for a specific site to understand how they are represented.
Action 1: assets
Action Input 1: site_name=MAIN
Observation 1: {"site_name": "MAIN", "total_assets": 5, "file_path": "/var/folders/wn/ddv2p1854tj06tv1h92t4hp00000gn/T/cbmdir/360fea44-6bb2-4f41-89d7-9a8313714589.json", "message": "found 5 assets for site_name MAIN. file_path contains a JSON array of Asset data"}
Thought 2: I now have the list of assets for site MAIN, which is stored in a file.
Action 2: jsonreader
Action Input 2: file_name=/var/folders/wn/ddv2p1854tj06tv1h92t4hp00000gn/T/cbmdir/360fea44-6bb2-4f41-89d7-9a8313714589.json
Observation 2: [{"site_name": "MAIN", "asssetnum": "B706_LC26"}, {"site_name": "MAIN", "asssetnum": "B012 SCWP 1"}, {"site_name": "MAIN", "asssetnum": "B710 FCU 1-39"}, {"site_name": "MAIN", "asssetnum": "B006_LC74"}, {"site_name": "MAIN", "asssetnum": "B006A AHU 2"}]
Thought 3: I now have the list of assets for site MAIN. I will provide five examples of how assets are represented.
Action 3: Finish
Action Input 3: Assets are represented with a site_name and an asssetnum. Here are five examples: 1. {"site_name": "MAIN", "asssetnum": "B706_LC26"} 2. {"site_name": "MAIN", "asssetnum": "B012 SCWP 1"} 3. {"site_name": "MAIN", "asssetnum": "B710 FCU 1-39"} 4. {"site_name": "MAIN", "asssetnum": "B006_LC74"} 5. {"site_name": "MAIN", "asssetnum": "B006A AHU 2"}

Question: How sites are represented? Give five examples.
Thought 1: I need to get the list of available sites to answer the question.
Action 1: sites
Action Input 1: {}
Observation 1: {"total_sites": 6, "file_path": "/var/folders/wn/ddv2p1854tj06tv1h92t4hp00000gn/T/cbmdir/f096b286-88b9-4817-8af6-a1783c205213.json", "message": "found 6 sites. file_path contains a JSON array of Site data"}
Thought 2: I now have the list of available sites, which is stored in a file.
Action 2: jsonreader
Action Input 2: file_name=/var/folders/wn/ddv2p1854tj06tv1h92t4hp00000gn/T/cbmdir/f096b286-88b9-4817-8af6-a1783c205213.json
Observation 2: [{"site_name": "HQ3SBY"}, {"site_name": "CNPANGU"}, {"site_name": "ARCMAIN"}, {"site_name": "AUSWEST"}, {"site_name": "TOR8200"}, {"site_name": "AUSEAST"}]
Thought 3: I now have the list of available sites.
Action 3: Finish
Action Input 3: Sites are represented by their names. Here are five examples: HQ3SBY, CNPANGU, ARCMAIN, AUSWEST, TOR8200.

Question: How sensors are represented? Give five examples.
Thought 1: I need to get the list of sensors for a specific asset to provide examples of how sensors are represented.
Action 1: sensors
Action Input 1: asssetnum=Chiller 1, site_name=MAIN
Observation 1: {"site_name": "MAIN", "asssetnum": "Chiller 1", "total_sensors": 12, "file_path": "/var/folders/wn/ddv2p1854tj06tv1h92t4hp00000gn/T/cbmdir/ed99f451-7e4e-4b7c-8bad-b56021a87332.json", "message": "found 12 sensors for asssetnum Chiller 1 and site_name MAIN. file_path contains a JSON array of Sensor data"}
Thought 2: I now have the list of sensors for Chiller 1 at site MAIN, which is stored in a file.
Action 2: jsonreader
Action Input 2: file_name=/var/folders/wn/ddv2p1854tj06tv1h92t4hp00000gn/T/cbmdir/ed99f451-7e4e-4b7c-8bad-b56021a87332.json
Observation 2: [{"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Supply Temperature"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Condenser Water Flow"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Liquid Refrigerant Evaporator Temperature"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Chiller % Loaded"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Chiller Efficiency"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Return Temperature"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Tonnage"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Condenser Water Supply To Chiller Temperature"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Power Input"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Run Status"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Setpoint Temperature"}, {"site_name": "MAIN", "asssetnum": "Chiller 1", "sensor_name": "Chiller 1 Schedule"}]
Thought 3: I now have the list of sensors for Chiller 1 at site MAIN.
Action 3: Finish
Action Input 3: Sensors are represented by their names. Here are five examples: Chiller 1 Supply Temperature, Chiller 1 Condenser Water Flow, Chiller 1 Liquid Refrigerant Evaporator Temperature, Chiller 1 Chiller % Loaded, Chiller 1 Chiller Efficiency.

Question: How asset is described? Give an example.
Thought 1: I need to get the description of an asset to answer the question.
Action 1: assetdescription
Action Input 1: site_name=MAIN, asssetnum=Chiller 6
Observation 1: {"asssetnum": "Chiller 6", "asset_type": "Chiller", "asset_location": "Building 020", "site_name": "MAIN"}
Thought 2: I now have the description of the asset Chiller 6 at site MAIN.
Action 2: Finish
Action Input 2: The asset Chiller 6 at site MAIN is described as follows: {"asssetnum": "Chiller 6", "asset_type": "Chiller", "asset_location": "Building 020", "site_name": "MAIN"}.

Question: How asset location is described? use MAIN site for B012 SCWP 1 asset as an example.
Thought 1: I need to get the asset description for B012 SCWP 1 at site MAIN to answer the question.
Action 1: assetdescription
Action Input 1: site_name=MAIN, asssetnum=B012 SCWP 1
Observation 1: {"asssetnum": "B012 SCWP 1", "asset_type": "Pump", "asset_location": "Building 012", "site_name": "MAIN", "asset_id": "B012 SCWP 1"}
Thought 2: I now have the asset description for B012 SCWP 1 at site MAIN.
Action 2: Finish
Action Input 2: The asset location for B012 SCWP 1 at site MAIN is described as "Building 012".
"""

SKYSPARK1Reflection = """


Previous Trial:
Question: Download sensor data for Chiller 6's Tonnage from last week at the MAIN site.
Thought 1: To download sensor data for Chiller 6's Tonnage from last week at the MAIN site, I need to follow these steps: 1. Get the current time to determine the date range for last week. 2. Retrieve the asset ID for Chiller 6 at the MAIN site. 3. Get the sensor data for the Tonnage sensor for Chiller 6 within the specified date range.
Action 1: currenttime
Action Input 1: {} 
{"current_time": "2024-12-04T01:29:49.006112+00:00"}
Observation 1: {"current_time": "2024-12-04T01:29:49.006112+00:00"}
Thought 2: I have the current time. Now I need to calculate the date range for last week.
Action 2: None
Action Input 2: None
Expecting value: line 1 column 1 (char 0)
Observation 2: Taking action: Issue while parsing Action
Thought 3: I have the current time. Now I need to calculate the date range for last week.
Action 3: None
Action Input 3: None
Expecting value: line 1 column 1 (char 0)
Observation 3: Taking action: Issue while parsing Action
Thought 4: I have the current time. Now I need to calculate the date range for last week.
Action 4: None
Action Input 4: None
Expecting value: line 1 column 1 (char 0)
Observation 4: Taking action: Issue while parsing Action

Reflection: I got stuck calculating the date range for last week, resulting in an invalid Action. I should have just figured out the date range from the current_time value

"""
