import os
import json
import re

source_dir_path = "Viveka/linear_experiment_2_NN_Probing/generated_completions"
merge_dir = os.path.join(source_dir_path, "merged_generations")
os.makedirs(merge_dir, exist_ok=True)

# Regex to extract both numeric parts in filenames like generated_0.0_3.2k.json
pattern = re.compile(r"generated_([0-9.]+)_([0-9.]+)k")

def extract_numbers(filename):
    match = pattern.search(filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return float("inf"), float("inf")

# Sort files by both numbers
files = sorted(
    [f for f in os.listdir(source_dir_path) if f.endswith(".json")],
    key=lambda name: extract_numbers(name)
)

merged_data = {}

for file in files:
    file_path = os.path.join(source_dir_path, file)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{file} is not a JSON object at the top level.")

        for question, content in data.items():
            if question not in merged_data:
                merged_data[question] = {
                    "prompt": content["prompt"],
                    "generated_answers": list(content["generated_answers"]),
                    "ground_truth_labels": list(content["ground_truth_labels"])
                }
            else:
                merged_data[question]["generated_answers"].extend(content["generated_answers"])
                merged_data[question]["ground_truth_labels"].extend(content["ground_truth_labels"])

# Always overwrite to avoid duplicates
output_path = os.path.join(merge_dir, "merged.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"Merged {len(files)} files into {output_path} with {len(merged_data)} questions total.")
