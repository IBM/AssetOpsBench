from IoTAgent.jsonreader import JSONReader
from IoTAgent.jsonreader.jsonwrapper import JSONWrapperFunctions
from IoTAgent.time.timewrapper import TimeWrapperFunctions
from IoTAgent.time.timetool import CurrentTimeJSONISO
from IoTAgent.bmstool import BMSHistory, BMSSites, BMSAssets, BMSSensors


def getTools():       

    bmsSites = BMSSites()
    bmsHistory = BMSHistory()
    bmsAssets = BMSAssets()
    bmsSensors = BMSSensors()

    jsonWrapper = JSONWrapperFunctions()
    jsonReaderTool = JSONReader(functions=jsonWrapper)

    fns = TimeWrapperFunctions()
    currenttimejsoniso = CurrentTimeJSONISO(functions=fns)

    tools = [
        bmsSites,
        bmsHistory,
        bmsAssets,      
        bmsSensors,
        jsonReaderTool,
        currenttimejsoniso,
    ]

    return tools
