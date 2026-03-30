from typing import Type
from langchain_core.tools import BaseTool
from reactxen.tools.jsonreader.jsonwrapper import JSONWrapperFunctions, JSONPropertyReturn
from pydantic import BaseModel, Field
import json
import tempfile
from pathlib import Path, PurePath
from uuid import uuid4

def getTempFilename():
    tmpdir = tempfile.gettempdir()
    tmppath = Path(tmpdir)
    basepath = Path("cbmdir")
    filename = str(uuid4())

    tmpdirpath = PurePath.joinpath(tmppath, basepath)

    tmpdirpath.mkdir(exist_ok=True)

    filepath = PurePath.joinpath(tmpdirpath, Path(filename + ".json"))

    return str(filepath)

class JSONReaderInputs(BaseModel):
    file_name: str = Field(description="name of JSON input file")


class JSONReader(BaseTool):
    """Tool to"""

    name: str = "jsonreader"
    description: str = (
        "Reads a JSON file, parses its content, and returns the parsed data."
    )
    args_schema: Type[BaseModel] = JSONReaderInputs
    response_format: str = "JSON"
    functions: JSONWrapperFunctions

    def _run(self, file_name: str) -> str:
        contents: any = self.functions.readFile(file_name)

        jsonContents = json.dumps(contents)

        return jsonContents

class JSONProperties(BaseTool):
    """Tool to"""

    name: str = "jsonproperties"
    description: str = (
        "Reads a JSON file, and returns a list of the properties contained in the JSON."
    )
    args_schema: Type[BaseModel] = JSONReaderInputs
    response_format: str = "JSON"
    functions: JSONWrapperFunctions

    def _run(self, file_name: str) -> str:
        contents: any = self.functions.properties(file_name)

        jsonContents = json.dumps(contents, default=custom_json)

        return jsonContents


def custom_json(obj):

    if isinstance(obj, JSONFileMergeMessage):

        return {
            "file_name_1": obj.file_name_1,
            "file_name_2": obj.file_name_2,
            "file_path": obj.file_path,
            "message": obj.message,
        }
    
    if isinstance(obj, JSONPropertyReturn):
        return { 
            "properties": ", ".join(obj.properties)
        }


class JSONFileMergeInputs(BaseModel):
    file_name_1: str = Field(description="name of first JSON input file")
    file_name_2: str = Field(description="name of second JSON input file")


class JSONFileMergeMessage:
    file_name_1: str
    file_name_2: str
    file_path: str
    message: str


class JSONFileMerge(BaseTool):
    """tool to merge two JSON files"""

    name: str = "jsonfilemerge"
    description: str = (
        "Merges the contents of two JSON files and returns the combined result."
    )
    args_schema: Type[BaseModel] = JSONFileMergeInputs
    response_format: str = "JSON"
    functions: JSONWrapperFunctions

    def _run(self, file_name_1: str, file_name_2: str) -> str:
        contents: any = self.functions.mergeFile(file_name_1, file_name_2)

        tmpfilename = getTempFilename()

        out = open(tmpfilename, "w")
        json.dump(contents, out, default=None)
        out.close()

        message = f"{file_name_1} and {file_name_2} have been merged into {tmpfilename}"

        retval = JSONFileMergeMessage()
        retval.file_name_1 = file_name_1
        retval.file_name_2 = file_name_2
        retval.file_path = tmpfilename
        retval.message = message

        return json.dumps(retval, default=custom_json)
