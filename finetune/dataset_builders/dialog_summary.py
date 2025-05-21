import json

input_json = 'samsum_1000.json'
output_json = 'samsum_train.json'

with open(input_json, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

output_data_lst = []

for line in input_data:
    item = {}
    
    item["instruction"] = "Summarize this dialog:"
    
    item["input"] = line["dialogue"]
    
    item["output"] = line["summary"]
    
    output_data_lst.append(item)

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)

print(f"Converted data has been saved to {output_json}")
