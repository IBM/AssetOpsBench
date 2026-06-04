#!/usr/bin/env python3
"""
Convert chiller9_tsad.csv to JSON format compatible with IoT MCP server.

This script reads the CSV file and converts it to a JSON array format
similar to motor_01.json, with each row becoming a JSON object containing
timestamp, asset_id, and sensor readings.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def clean_field_name(field: str) -> str:
    """Keep field names as-is to match existing IoT JSON format.

    Example: "Chiller 9 Return Temperature" stays as "Chiller 9 Return Temperature"
    """
    # Keep the field name unchanged to match chiller_6.json format
    return field


def convert_value(value: str) -> Any:
    """Convert string value to appropriate type (float, int, or string)."""
    if not value or value.strip() == "":
        return None

    try:
        # Try to convert to float first
        float_val = float(value)
        # If it's a whole number, convert to int
        if float_val.is_integer() and "." not in value:
            return int(float_val)
        return float_val
    except ValueError:
        # Keep as string if conversion fails
        return value


def csv_to_json(csv_path: Path, json_path: Path, asset_id: str = "Chiller 9") -> None:
    """Convert CSV file to JSON format.

    Args:
        csv_path: Path to input CSV file
        json_path: Path to output JSON file
        asset_id: Asset identifier to use in JSON records
    """
    records: List[Dict[str, Any]] = []

    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            record = {"asset_id": asset_id}

            for key, value in row.items():
                # Handle timestamp field specially
                if key == "Timestamp":
                    record["timestamp"] = value
                else:
                    # Clean field name and convert value
                    clean_key = clean_field_name(key)
                    record[clean_key] = convert_value(value)

            records.append(record)

    # Write JSON file with pretty formatting
    with open(json_path, "w", encoding="utf-8") as jsonfile:
        json.dump(records, jsonfile, indent=2)

    print(
        f"✓ Converted {len(records)} records from {csv_path.name} to {json_path.name}"
    )
    print(f"  Output file: {json_path}")


def main():
    """Main function to convert chiller CSV files to JSON."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Define files to convert
    files_to_convert = [
        "chiller9_tsad.csv",
        "chiller9_finetuning_small.csv",
        "chiller9_annotated_small_test.csv",
    ]

    for csv_filename in files_to_convert:
        csv_file = script_dir / csv_filename
        json_file = script_dir / csv_filename.replace(".csv", ".json")

        # Check if CSV file exists
        if not csv_file.exists():
            print(f"⚠ Skipping: CSV file not found: {csv_file}")
            continue

        # Convert CSV to JSON
        print(f"\nConverting {csv_file.name} to JSON format...")
        csv_to_json(csv_file, json_file, asset_id="Chiller 9")

        # Show sample of the output
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Sample record (first entry):")
            print(json.dumps(data[0], indent=2))
            print(f"Total records: {len(data)}")

    print("\n✅ All conversions complete!")


if __name__ == "__main__":
    main()

# Made with Bob
