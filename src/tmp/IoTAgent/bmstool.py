import couchdb3
import json 
import logging

from typing import Optional, Type
from typing import Optional
from pydantic import BaseModel, Field
import json
from typing import Type
from langchain_core.tools import BaseTool
from iotagent.demo.tool import getTempFilename
from datetime import datetime

client = couchdb3.Server("http://admin:password@localhost:5984/")

db = client.get("main")

print(db)

logger: logging.Logger = logging.getLogger(__name__)


def custom_json(obj):

    if isinstance(obj, SitesMessage):
        return {
            # "sites": '[' + ', '.join(obj.sites) + ']',
            'sites': obj.sites
        }
    
    if isinstance(obj, AssetMessage):
        return {
            'site_name': obj.site_name,
            'total_assets': obj.total_assets,
            'file_path': obj.file_path,
            'message': obj.message,
        }

    # if isinstance(obj, SensorDescription):
    #     return {
    #         'sensor_name': obj.sensor_name,
    #         'name': obj.name,
    #     }
    
    if isinstance(obj, SensorMessage):
        return {
           'site_name': obj.site_name,
            'assetnum': obj.assetnum,
            'total_sensors': obj.total_sensors,
            # 'sensor_list': custom_json(obj.sensor_list),
            'file_path': obj.file_path,
            'message': obj.message
        }
    
    if isinstance(obj, HistoryMessage):
        return {   
            'site_name': obj.site_name,
            'assetnum': obj.assetnum,
            'total_observations': obj.total_observations,
            'start': obj.start,
            'final': obj.final,
            'file_path': obj.file_path,
            'message': obj.message,
        }

    raise TypeError(f"Cannot serialize object of {type(obj)}")

SITES = [
    'MAIN'
]

# ASSETS = [ 'Chiller 4' ]

SENSORS = {
    'CQPA AHU 1': [
        'CQPA AHU 1 Pre Heating Temp',
        'CQPA AHU 1 Cooling Valve %',
        'CQPA AHU 1 Supply Relative Humidity Setpoint %',
        'CQPA AHU 1 Preheat Valve %',
        'CQPA AHU 1 Return Air Temperature',
        'CQPA AHU 1 Supply Relative Humidity %',
        'CQPA AHU 1 Zone Relative Humidity %',
        'CQPA AHU 1 Occupied Command',
        'CQPA AHU 1 Setpoint Temperature',
        'CQPA AHU 1 Supply Fan Output %',
        'CQPA AHU 1 Supply Air Temperature',
        'CQPA AHU 1 Supply Fan Status',
        'CQPA AHU 1 Power (Calc)',
        'CQPA AHU 1 Schedule',
        'CQPA AHU 1 Mixed Air Temperature',
        'CQPA AHU 1 Humidifier Valve %',
    ],
    'CQPA AHU 2B': [
        'CQPA AHU 2B Cooling Valve %',
        'CQPA AHU 2B Supply Fan Output %',
        'CQPA AHU 2B Pre Heating Temp',
        'CQPA AHU 2B Return Air Temperature',
        'CQPA AHU 2B Mixed Air Temperature',
        'CQPA AHU 2B Supply Air Temperature',
        'CQPA AHU 2B Economizer Mode',
        'CQPA AHU 2B Setpoint Temperature',
        'CQPA AHU 2B Power (Calc)',
        'CQPA AHU 2B Supply Fan Status',
        'CQPA AHU 2B Supply Fan Current',
        'CQPA AHU 2B Duct Static Pressure',
        'CQPA AHU 2B Preheat Valve %',
        'CQPA AHU 2B Schedule',
        'CQPA AHU 2B Static Pressure Setpoint',
        'CQPA AHU 2B Occupied Command',
    ],
    'Chiller 4': [
        "Chiller 4 Liquid Refrigerant Evaporator Temperature",
        "Chiller 4 Condenser Water Supply To Chiller Temperature",
        "Chiller 4 Return Temperature",
        "Chiller 4 Setpoint Temperature",
        "Chiller 4 Chiller % Loaded",
        "Chiller 4 Supply Temperature",
        "Chiller 4 Condenser Water Flow",
        "Chiller 4 Tonnage",
        "Chiller 4 Power Input",
        "Chiller 4 Chiller Efficiency",
    ],

    'Chiller 6': [
        'Chiller 6 Condenser Water Return To Tower Temperature',
        'Chiller 6 Chiller Efficiency',
        'Chiller 6 Tonnage',
        'Chiller 6 Supply Temperature',
        'Chiller 6 Return Temperature',
        'Chiller 6 Run Status',
        'Chiller 6 Condenser Water Flow',
        'Chiller 6 Schedule',
        'Chiller 6 Power Input',
        'Chiller 6 Chiller % Loaded',
        'Chiller 6 Liquid Refrigerant Evaporator Temperature',
        'Chiller 6 Setpoint Temperature',
    ],

    'Chiller 9': [ 
        'Chiller 9 Setpoint Temperature',
        'Chiller 9 Supply Temperature',
        'Chiller 9 Tonnage',
        'Chiller 9 Run Status',
        'Chiller 9 Return Temperature',
        'Chiller Efficiency',
        'Chiller 9 Schedule',
        'Chiller 9 Power Input',
        'Chiller 9 Chiller % Loaded',
        'Chiller 9 Condenser Water Flow',
        'Chiller 9 Liquid Refrigerant Evaporator Temperature',
        'Chiller 9 Condenser Water Supply To Chiller Temperature',
    ],

    'Chiller 3': [ 
        'Chiller 3 Condenser Water Flow',
        'Chiller 3 Chiller Efficiency',
        'Chiller 3 Liquid Refrigerant Evaporator Temperature',
        'Chiller 3 Run Status',
        'Chiller 3 Tonnage',
        'Chiller 3 Chiller % Loaded',
        'Chiller 3 Supply Temperature',
        'Chiller 3 Condenser Water Supply To Chiller Temperature',
        'Chiller 3 Schedule',
        'Chiller 3 Setpoint Temperature',
        'Chiller 3 Power Input',
        'Chiller 3 Return Temperature',
    ]
}

# class SensorDescription:
#     sensor_name: str
#     name: str

class BMSAssetsInputs(BaseModel):
    site_name: str = Field(description="site or location of the assets")


class BMSAssetDescriptionInputs(BaseModel):
    site_name: str = Field(description="site or location of the assets")
    asset_name: str = Field(description="name of asset (not Tag/Reference)")


class AssetMessage:
    site_name: str
    total_assets: int
    file_path: str
    message: str


class BMSAssets(BaseTool):
    """Tool to fetch assets at a given site from a BMS datastore"""

    name: str = "assets"
    description: str = (
        "Returns a list of assets for a given site. Each asset includes an id and a name."
    )
    args_schema: Type[BaseModel] = BMSAssetsInputs
    response_format: str = "JSON"

    def _run(self, site_name: str) -> str:
        
        if site_name not in SITES:
            raise ValueError(f'unknown site "{site_name}"')
        
        tmpfilename = getTempFilename()

        out = open(tmpfilename, "w")
        assets = list(SENSORS.keys())
        json.dump(assets, out)#, default=custom_json)
        out.close()

        total_assets = len(SENSORS)
        message = f"found {total_assets} assets for site_name {site_name}. file_path contains a JSON array of Asset data"

        retval = AssetMessage()
        retval.site_name = site_name
        retval.total_assets = total_assets
        retval.file_path = tmpfilename
        retval.message = message

        return json.dumps(retval, default=custom_json)


class BMSAssetDescription(BaseTool):
    """Tool to fetch description such as type and location information of asset at a given site using asset name from a BMS datastore"""

    name: str = "assetdescription"
    description: str = (
        "Return an asset description for a given site and an asset. Asset description includes asset id, asset type and location information."
    )
    args_schema: Type[BaseModel] = BMSAssetDescriptionInputs
    response_format: str = "JSON"

    def _run(self, site_name: str, asset_name: str) -> str:
        # Fetch the asset description (assuming it returns a single AssetDescription object)
        asset_description: AssetDescription = self.BMS_functions.asset_description(
            site_name=site_name, asset_name=asset_name
        )

        # Convert the single asset description to JSON using custom serialization
        return json.dumps(asset_description, default=self.custom_json)

    def custom_json(self, obj):
        """Custom JSON serialization for complex objects like AssetDescription"""
        if isinstance(obj, AssetDescription):
            # Serialize the AssetDescription object into a dictionary
            return {
                "asset_name": obj.asset_name,
                "asset_type": obj.asset_type,
                "asset_location": obj.asset_location,
                "site_name": obj.site_name,
                "asset_id": obj.asset_id,
                # Add other fields if needed
            }
        raise TypeError(f"Type {type(obj)} not serializable")


class SitesMessage:
    sites: str

class BMSSites(BaseTool):
    """Tool to fetch sites from a BMS datastore"""

    name: str = "sites"
    description: str = "Retrieves a list of sites. Each site is represented by a name."
    response_format: str = "JSON"

    def _run(self, args: None = None) -> str:

        retval = SitesMessage()
        retval.sites = SITES

        return json.dumps(retval, default=custom_json)


class BMSHistoryInputs(BaseModel):
    site_name: str = Field(description="site or location of the assets")
    assetnum: str = Field(
        description="Asset Number"
    )
    start: str = Field(description="start datetime to return")
    # sensor_name_list: Optional[str] = Field(
    #     description="name of sensor(s) for which to return history. If None return all sensors"
    # )
    final: Optional[str] = Field(
        description="final datetime to return.  If None, return only the values from the start datetime"
    )


class HistoryMessage:
    site_name: str = Field(description="site or location of the assets")
    assetnum: str
    total_observations: int
    start: str
    final: str
    file_path: str = Field(description="path to file of sensor information")
    message: str = Field(description="response message")


class BMSHistory(BaseTool):
    """Tool to return sensor history for an asset at a site"""

    name: str = "history"
    description: str = (
        "Returns a list of historical sensor values for the specified asset(s) at a site within a given time range (start to final)."
    )
    args_schema: Type[BaseModel] = BMSHistoryInputs
    response_format: str = "JSON"

    def _run(
        self,
        site_name: str,
        assetnum: str,
        start: str,
        final: Optional[str] = None,
        # sensor_name_list: Optional[str] = None,
    ) -> str:
        
        if assetnum not in SENSORS:
            raise ValueError(f'no such assetnum {assetnum}')
        
        selector = {
            "asset_id": assetnum,
            "timestamp": {
                "$gte": datetime.fromisoformat(start).isoformat()
            }
        }

        if final is not None:
            selector["timestamp"]["$lt"] = datetime.fromisoformat(final).isoformat()

            strt = datetime.fromisoformat(start).isoformat()
            fin = datetime.fromisoformat(final).isoformat()

            if strt >= fin:
                raise ValueError('start >= final')

        # fields = None
        # assetSensorProps = SENSORS[assetnum]
        # if sensor_name_list is not None:
        #     fields = ['_id']
        #     sensorList = sensor_name_list.split(',')

        #     for nonTrimmedSensor in sensorList:
        #         sensor = nonTrimmedSensor.strip()
                 
        #         property = None
        #         for entry in assetSensorProps:
        #             if entry['sensor_name'] == sensor:
        #                 property = entry['sensor_name']
        #                 break

        #         if property is None:
        #             raise ValueError(f'sensor {sensor} not found on assetnum {assetnum}')

        #         fields.append(property)

        # print('selecting:\n', json.dumps(selector, indent=2))
        # print(f'fields = {fields}')

        logger.critical(f'******* selector ={json.dumps(selector, indent=2)}')
        res = db.find(selector, limit=100000, sort=[{"asset_id": "asc"}, {"timestamp": "asc"}]) 
                # limit: int = 25, skip: int = 0, sort: List[Dict] = None, fields: List[str] = None, use_index: Union[str, List[str]] = None, conflicts: bool = False, r: int = 1, bookmark: str = None, update: bool = True, stable: bool = None, execution_stats: bool = False, partition: str = None) ‑> Dict

        # print('res =', json.dumps(res, indent=2))


        tmpfilename = getTempFilename()

        docs = res['docs']
        # for doc in docs:
        #     doc['timestamp'] = doc['_id']
            
        out = open(tmpfilename, "w")
        json.dump(docs, out)
        out.close()

        total_observations = len(docs)
        message = f"found {total_observations} observations. file_path contains a JSON array of Observation data"

        retval = HistoryMessage()
        retval.site_name = site_name
        retval.assetnum = assetnum
        retval.total_observations = total_observations
        retval.start = start
        retval.final = final
        # retval.sensor_name = sensor_name
        retval.file_path = tmpfilename
        retval.message = message

        return json.dumps(retval, default=custom_json)


class SensorMessage:
    site_name: str = Field(description="site or location of the assets")
    assetnum: str = Field(description="asset number")
    total_sensors: int = (Field(description="total number of sensoors"),)
    file_path: str = Field(description="path to file of sensor information")
    message: str = Field(description="response message")


class BMSSensorsInputs(BaseModel):
    site_name: str = Field(description="site or location of the assets")
    assetnum: str = Field(description="assetnum of asset")


class BMSSensors(BaseTool):
    """given a site and asset, list available metadata for the asset"""

    name: str = "sensors"
    description: str = (
        "Lists the sensors available for a specified asset at a given site."
    )
    args_schema: Type[BaseModel] = BMSSensorsInputs
    response_format: str = "json"

    def _run(self, site_name: str, assetnum: str) -> str:

        if site_name not in SITES:
            raise ValueError(f'unknown site "{site_name}"')
        
        if assetnum not in SENSORS:
            raise ValueError(f'unknown assetnum "{assetnum}"')

        sensors = SENSORS[assetnum]

        tmpfilename = getTempFilename()

        out = open(tmpfilename, "w")
        json.dump(sensors, out, default=custom_json)
        out.close()

        total_sensors = len(sensors)
        message = f"found {total_sensors} sensors for assetnum {assetnum} and site_name {site_name}. file_path contains a JSON array of Sensor data"

        retval = SensorMessage()
        retval.site_name = site_name
        retval.assetnum = assetnum
        retval.total_sensors = total_sensors
        retval.file_path = tmpfilename
        retval.message = message

        out = json.dumps(retval, default=custom_json)
        
        return out
