import json
import csv

input_path = "data/problems_data.jsonl"
output_path = "data/problems_data.csv"

with open(input_path, "r", encoding="utf-8") as fin,\
     open(output_path, "w", newline='', encoding="utf-8") as fout:
         
    writer = None
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        if writer is None:
            headers = list(obj.keys())
            writer = csv.DictWriter(fout, fieldnames=headers)
            writer.writeheader()
        writer.writerow(obj)
