import json

# Replace with your actual filenames
input_file = 'senarios_10_asset_class.jsonl'
output_file = 'failure_mapping_senarios.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Load the JSON object from the current line
        data = json.loads(line)
        
        # Remove the 'answer' key if it exists
        data.pop('answer', None)
        
        # Write the modified dictionary back as a line
        outfile.write(json.dumps(data) + '\n')