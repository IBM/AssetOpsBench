import argparse
import json
import pathlib
from typing import List, Iterator, Dict, Any
from pydantic import ValidationError

from scenarios import Scenario

def read_json_file(file_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Read and parse JSON file, handling both arrays and single objects."""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {str(e)}")

def read_jsonl_file(file_path: pathlib.Path) -> Iterator[Dict[str, Any]]:
    """Read and parse JSONL file line by line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} in {file_path}: {str(e)}")

def validate_scenario(data: Dict[str, Any], context: str = "") -> List[str]:
    """Validate a single scenario object against the Scenario model."""
    errors = []
    try:
        # Convert id to string if it's an integer (based on your examples)
        if 'id' in data and isinstance(data['id'], int):
            data['id'] = str(data['id'])
        
        Scenario(**data)
    except ValidationError as e:
        for error in e.errors():
            field = " → ".join(str(loc) for loc in error['loc'])
            msg = error['msg']
            errors.append(f"{context} - Field '{field}': {msg}")
    except Exception as e:
        errors.append(f"{context} - Unexpected error: {str(e)}")
    return errors

def validate_file(file_path: pathlib.Path) -> List[str]:
    """Validate all scenarios in a file."""
    errors = []
    try:
        if file_path.suffix == '.jsonl':
            # Process JSONL file
            for line_num, data in enumerate(read_jsonl_file(file_path), 1):
                context = f"{file_path}:{line_num}"
                errors.extend(validate_scenario(data, context))
        else:
            # Process JSON file
            data_list = read_json_file(file_path)
            for idx, data in enumerate(data_list, 1):
                context = f"{file_path}[{idx}]"
                errors.extend(validate_scenario(data, context))
    except Exception as e:
        errors.append(f"{file_path} - Failed to process file: {str(e)}")
    return errors

def find_json_files(directory: pathlib.Path) -> List[pathlib.Path]:
    """Find all JSON and JSONL files in a directory."""
    json_files = list(directory.glob("*.json"))
    jsonl_files = list(directory.glob("*.jsonl"))
    return json_files + jsonl_files

def main():
    parser = argparse.ArgumentParser(
        description='Validate JSON/JSONL files against Scenario schema.'
    )
    parser.add_argument(
        'path', 
        help='Path to file or directory containing JSON/JSONL files'
    )
    args = parser.parse_args()

    path = pathlib.Path(args.path)
    all_errors = []

    if path.is_file():
        print(f"Validating file: {path}")
        all_errors.extend(validate_file(path))
    elif path.is_dir():
        print(f"Validating files in directory: {path}")
        json_files = find_json_files(path)
        if not json_files:
            print(f"No JSON/JSONL files found in {path}")
            return
            
        for file_path in json_files:
            print(f"Validating: {file_path.name}")
            all_errors.extend(validate_file(file_path))
    else:
        print(f"Error: Path '{args.path}' does not exist.")
        return

    # Print all errors
    for error in all_errors:
        print(f"ERROR: {error}")

    # Summary
    if all_errors:
        print(f"\nValidation failed with {len(all_errors)} error(s)")
        return 1
    else:
        print("\nAll files passed validation!")
        return 0

if __name__ == '__main__':
    exit(main())