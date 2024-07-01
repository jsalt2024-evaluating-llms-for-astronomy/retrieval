import json

path = "../data/multi_paper.json"
with open(path, 'r') as f:
    data = json.load(f)

# Print number of elements in the data dictionary
print(len(data))

# Go through and remove any dictionary elements where the dictionary corresponding to the key has an empty list for arxiv
keys_to_remove = []
for key, value in data.items():
    if len(value['arxiv']) == 0:
        keys_to_remove.append(key)

# Print the number of keys to remove
print(len(keys_to_remove))

# Remove the keys
for key in keys_to_remove:
    data.pop(key)

# Print the number of elements in the data dictionary
print(len(data))

# Save the data dictionary back to the file
with open(path, 'w') as f:
    json.dump(data, f)