import json

from openai import OpenAI
import os
from collections import deque
from dotenv import load_dotenv  
load_dotenv()


# Initialize OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=api_key)

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


# Function to generate embeddings using OpenAI API
def generate_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    return response['data'][0]['embedding']

# Read the original JSON file
with open('restructured_categories.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize a queue for breadth-first traversal
queue = deque([(data['categories'], None)])
unique_id = 0

# Traverse and augment JSON
while queue:
    current_level, parent_category = queue.popleft()
    
    for category in current_level:
        # Assign unique ID
        category['unique_id'] = unique_id
        print(unique_id)
        unique_id += 1

        # Generate embedding text
        embedding_text = f"{category['category_name']} {category['parent_category']}"

        # Generate and assign embedding
        category['embedding'] = get_embedding(embedding_text)

        # If there are subcategories, add them to the queue
        if category['subcategories']:
            queue.append((category['subcategories'], category['category_name']))

# Save augmented JSON to a new file
with open('augmented_categories.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)
