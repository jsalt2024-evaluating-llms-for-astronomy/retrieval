import json

path = "/Users/charlesoneill/retrieval/data/multi_paper.json"

with open(path, 'r') as f:
    data = json.load(f)

# Remove any data where the arxiv field is empty list
data = [{k: v} for k, v in data.items() if v['arxiv']]
print(len(data))

# Save back to multi_paper.json
with open(path, 'w') as f:
    json.dump(data, f, indent=4)

print("Done!")