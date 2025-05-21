import json

input_json = 'sqlgen_1000.json'
output_json = 'sqlgen_train.json'

with open(input_json, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

output_data_lst = []

for line in input_data:
    item = {}
    
    item["instruction"] = (
        "Please convert the provided natural language query into an SQL query, "
        "taking into account the structure of the database defined by the accompanying CREATE statement:\n"
    )
    
    item["input"] = (
        f"## Natural Language Query:\n{line['question']}\n"
        f"## Context:\n{line['context']}\n"
    )
    
    item["output"] = line["answer"]
    
    output_data_lst.append(item)

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)

print(f"Converted data has been saved to {output_json}")
