import json
import pandas as pd

# Load the JSON data
with open('augmented_categories.json', 'r') as file:
    data = json.load(file)

# Function to recursively extract unique_id and embedding
def extract_data(subcategories):
    extracted = []
    for sub_item in subcategories:
        if 'unique_id' in sub_item and 'embedding' in sub_item:
            unique_id = sub_item['unique_id']
            embedding = sub_item['embedding']
            extracted.append({'unique_id': unique_id, 'embedding': embedding})
            # Remove the embedding from the original JSON
            del sub_item['embedding']
        # Check for further nested subcategories
        if 'subcategory' in sub_item:
            extracted.extend(extract_data(sub_item['subcategories']))
    return extracted

def recursive_items(item, extracted_data):

    print(item['unique_id'])
    unique_id = item['unique_id']
    embedding = item['embedding']
    extracted_data.append({'unique_id': unique_id, 'embedding': embedding})
    for item in item["subcategories"]:
        extracted_data = recursive_items(item, extracted_data)
        
    return extracted_data

# Start extraction from the top-level categories
extracted_data = []
for item in data['categories']:  # Assuming 'categories' is the key containing the list
    #print(item['unique_id'])
    recursive_items(item, extracted_data)
    #extracted_data.extend(extract_data(item['subcategories']))

# Create a DataFrame
df = pd.DataFrame(extracted_data)

df = df.sort_values(by="unique_id")
# Save the DataFrame to a CSV file
df.to_csv('unique_id_embeddings.csv', index=False)

# Save the modified JSON back to a file
with open('modified_augmented_categories.json', 'w') as file:
    json.dump(data, file, indent=4)