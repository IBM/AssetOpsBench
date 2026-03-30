import json

input_file = 'updated_scenarios.jsonl'
output_file = 'final_scenarios.jsonl'

# Define the exact keys we want to keep in the final output
required_keys = [
    "id", "text", "type", "category", "deterministic", 
    "characteristic_form", "group", "entity", "note"
]

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # 1. Load the current line
        data = json.loads(line)
        
        # 2. Add/Update the specific fields you requested
        data['type'] = 'phm'
        data['deterministic'] = True  # Using boolean True for JSON 'true'
        data['group'] = 'phm'
        data['entity'] = '' 
        data['note'] = ''
        
        # 3. Create a new dictionary with only the selected keys
        # We use .get(key, "") to ensure the key exists even if missing in source
        filtered_data = {key: data.get(key, "") for key in required_keys}
        
        # Ensure 'id' remains an integer if possible
            
        # 4. Write to the new file
        outfile.write(json.dumps(filtered_data) + '\n')

print(f"Transformation complete. File saved as {output_file}")