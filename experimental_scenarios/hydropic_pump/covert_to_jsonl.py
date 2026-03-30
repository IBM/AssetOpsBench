import json

# Replace 'scenario_modified.json' and 'hydrolic_pump_utterance.jsonl' with your filenames
with open('scenario_modified.json', 'r') as f:
    data = json.load(f)

with open('hydrolic_pump_utterance.jsonl', 'w') as f:
    for entry in data:
        # Create a new dictionary where list values are joined into strings
        processed_entry = {
            k: ", ".join(map(str, v)) if isinstance(v, list) else v 
            for k, v in entry.items()
        }
        
        # Write the processed dictionary as a single line
        f.write(json.dumps(processed_entry) + '\n')