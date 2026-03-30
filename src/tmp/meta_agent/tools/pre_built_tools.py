from reactxen.tools.jsonreader.jsonreader import JSONReader, JSONProperties
from reactxen.tools.jsonreader.jsonwrapper import JSONWrapperFunctions
from typing import Dict


# Tool creation function
def create_json_reader() -> JSONReader:
    """Create and return a JSONReader instance."""
    fns = JSONWrapperFunctions()
    return JSONReader(functions=fns)

def create_json_properties() -> JSONProperties:
    """Create and return a JSONProperties instance."""
    fns = JSONWrapperFunctions()
    return JSONProperties(functions=fns)


# Centralized function to load all prebuilt tools
def load_prebuilt_tools() -> Dict[str, object]:
    """
    Load prebuilt tools and return them as a dictionary.
    
    The dictionary keys are tool names, and values are the corresponding tool instances.
    """
    tools = {
        "JSONReader": create_json_reader(),
        # "JSONProperties": create_json_properties(),
        # Add more tools here as needed
    }
    return tools
