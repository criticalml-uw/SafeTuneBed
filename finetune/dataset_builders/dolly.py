import json

input_jsonl = '/Users/saadhossain/research_ws/SafeTuneBed/src/dataset_builders/databricks-dolly-15k-no-safety.jsonl'
output_json = 'dolly.json'

output_data_lst = []

with open(input_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            json_line = json.loads(line)

            item = {}
            item["instruction"] = json_line["instruction"]
            item["input"] = json_line["context"]
            item["output"] = json_line["response"]
            
            output_data_lst.append(item)

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)

print(f"Converted data has been saved to {output_json}")
