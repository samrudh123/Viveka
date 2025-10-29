import json
from pathlib import Path


# Input files
file1 = Path("Viveka/linear_experiment_2_NN_Probing/generated_completions/generated_0.0_3.2k.json")
file2 = Path("Viveka/linear_experiment_2_NN_Probing/generated_completions/generated_3.2_6.4k.json")

# Output file
merged_file = Path("generated_completions/merged.json")

# Always start with empty merged file to avoid duplicates
if merged_file.exists():
    merged_file.unlink()

# Load both JSON files
with open(file1, "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open(file2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

# Merge them (keys from data2 overwrite data1 if same question)
merged_data = {**data1, **data2}

# Save merged JSON
with open(merged_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"Merged {len(merged_data)} questions into {merged_file}")
