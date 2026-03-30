from langchain_core.tools import BaseTool
from reactxen.tools.time.timewrapper import TimeWrapperFunctions

import json
import pendulum

class CurrentTimeJSONISO(BaseTool):
    """Tool to return current date time"""

    # """Tool to return current time in JSON ISO format"""

    name: str = "currentdatetime"
    description: str = "Provides the current date time as a JSON object."
    # name: str = "currenttimejsoniso"
    # description: str = "return the current time in ISO format as a JSON object"
    response_format: str = "JSON"
    functions: TimeWrapperFunctions

    def _run(self, args: None = None) -> str:

        nowISO: str = self.functions.currentTime()
        nowDate: str = nowISO.split('T')[0]
        nowTime: str = nowISO.split('T')[1].split('.')[0]

        # Create a descriptive string with the current date and time
        description: str = f"Today's date is {nowDate} and time is {nowTime}."
        
        obj = {
            'currentDateTime': nowISO,
            'currentDateTimeDescription': description
        }

        retval = json.dumps(obj)

        return retval

class CurrentTimeEnglish(BaseTool):
    """Tool to return current time in English"""

    name: str = "currenttimeenglish"
    description: str = "return the current time in English text"
    response_format: str = "content_and_artifact"
    functions: TimeWrapperFunctions

    def _run(self, args: None = None) -> str:

        nowISO: str = self.functions.currentTime()

        obj = {
            'current_time': nowISO
        }

        retval = json.dumps(obj)

        dt = pendulum.parse(nowISO)

        eng = dt.to_datetime_string()

        return (eng, retval)
