import ast
import json


def json_parser(input_string):
    input_string = input_string.strip()
    input_string = input_string.replace("```json", "").replace("```", "")
    python_dict = ast.literal_eval(input_string)
    json_string = json.dumps(python_dict)
    json_dict = json.loads(json_string)

    if isinstance(json_dict, dict) or isinstance(json_dict, list):
        return json_dict

    raise "Invalid JSON response"
