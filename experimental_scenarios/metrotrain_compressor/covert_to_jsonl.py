import json

# Replace 'input.json' and 'output.jsonl' with your filenames
with open('scenario_revised.json', 'r') as f:
    data = json.load(f)

with open('compressor_utterance.jsonl', 'w') as f:
    for entry in data:
        f.write(json.dumps(entry) + '\n')