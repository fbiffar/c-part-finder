#run code with : streamlit run e_shop_2.py

# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
import requests
import json
from dotenv import load_dotenv  
import os
import requests
from PIL import Image
from googlesearch import search
import csv
import streamlit as st 
import pandas as pd

def get_image_from_finder():
    # Set up the app title
    st.title("Image Upload and Segmentation Setup")

    # File uploader for the image
    uploaded_file = st.file_uploader("Upload an image of machine", type=["png", "jpg", "jpeg"])

    # Row and column input fields
    rows = st.number_input("Enter the number of rows for segmentation:", min_value=1, max_value=100, step=1, format="%d")
    cols = st.number_input("Enter the number of columns for segmentation:", min_value=1, max_value=100, step=1, format="%d")

    # Process the uploaded file
    if uploaded_file:
        # Open the uploaded image
        image_pl = Image.open(uploaded_file)

        # Convert the image to a NumPy array (RGB format)
        image = np.array(image_pl)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Confirm and store the inputs
        if st.button("Confirm"):
            st.success(f"Image successfully uploaded and segmentation set to {rows} rows and {cols} columns.")

        return image, rows, cols    
# Function to display an image using Matplotlib
def display_image(img, title="Image"):
    plt.figure(figsize=(8, 8))
    if len(img.shape) == 3:  # Color image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:  # Grayscale image
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to divide the image into a grid
def divide_image_into_tiles(image, rows, cols):
    h, w = image.shape[:2]
    tile_h, tile_w = h // rows, w // cols
    tiles = []
    for i in range(rows):
        for j in range(cols):
            y_start, x_start = i * tile_h, j * tile_w
            y_end, x_end = y_start + tile_h, x_start + tile_w
            tile = image[y_start:y_end, x_start:x_end]
            tiles.append(tile)
    return tiles, tile_h, tile_w

# Function to overlay grid lines on the image
def overlay_grid(image, rows, cols):
    img_with_grid = image.copy()
    h, w = img_with_grid.shape[:2]
    tile_h, tile_w = h // rows, w // cols
    for i in range(1, rows):
        cv2.line(img_with_grid, (0, i * tile_h), (w, i * tile_h), (0, 255, 0), 2)
    for j in range(1, cols):
        cv2.line(img_with_grid, (j * tile_w, 0), (j * tile_w, h), (0, 255, 0), 2)
    return img_with_grid

# Function to arrange tiles in a single grid layout
def arrange_tiles_in_grid(tiles, rows, cols):
    tile_h, tile_w = tiles[0].shape[:2]
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        i, j = divmod(idx, cols)
        grid[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w] = tile
    return grid

# Function to encode an image to base64
def encode_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# Function to send image data to OpenAI API
def send_to_openai_api(encoded_image, api_endpoint, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
     
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please identify any machine components in this image. No rigorous explanation. List the components seperated by commas, if there are no machine parts found please type 'No machine parts found'."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300  
    }
    response = requests.post(api_endpoint, headers=headers, json=data)
    return response.json()

def print_response_and_store (response, idx):
    # Extract and print only the content
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "No content available")
    print(f"Tile {idx + 1} Response: {content}")
    return [[element.strip(), idx +1] for element in content.split(",") if element.strip()]
    
 


def search_bossard(component_name):
    query = f"{component_name} site: www.bossard.com"
    try:
        # Use next() to get the first result from the generator
        result = next(search(query, num_results=1, lang='de'), None)
        if result:
            return result
        else:
            return "Not available"
    except Exception as e:
        print(f"Error searching for '{component_name}': {e}")
        return "Not available"
    
# Function to store the component and its URL in the CSV file immediately
def store_in_csv(file_path,part_id, component, url, tile_id):
    if component != 'No machine parts found' and url != "https://www.bossard.com/ch-en/" and url != "Not available" and url != "https://www.bossard.com/": 
        with open(file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([part_id, component, url, tile_id])


def dialogue_return_csv(csv_file_path):
    # Notify the user
    st.success(f"CSV file generated: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path)
    st.subheader("Generated Table from CSV")
    st.dataframe(df)





#run code with : streamlit run e_shop_2.py








image, rows, cols = get_image_from_finder()
display_image(image, "Original Image")
tiles, tile_h, tile_w = divide_image_into_tiles(image, rows, cols)

load_dotenv()

# Access environment variables
api_key = os.getenv("OPENAI_API_KEY")
api_endpoint = os.getenv("OPENAI_API_ENDPOINT")

all_tile_elements = []
for idx, tile in enumerate(tiles):
        encoded_tile = encode_to_base64(tile)
        response = send_to_openai_api(encoded_tile, api_endpoint, api_key)
        all_tile_elements.extend(print_response_and_store(response, idx))
        #break
# Print the combined list of all elements
print("All Elements from Tiles:", all_tile_elements)

availability = {}
# Initialize CSV file with header
csv_file_path = "c-part-finder/result_csv/machine_parts_live.csv"
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Part ID","Machine Part", "URL", "Tile ID"])

for part_id, component in enumerate(all_tile_elements, start=1): 
    component_name, tile_id = component
    url = search_bossard(component_name)
    availability[component_name] = url
    print(f"Part ID: {part_id}, Component: {component_name} - URL: {url}")
    store_in_csv(csv_file_path, part_id, component_name, url, tile_id) 
    #break 

dialogue_return_csv(csv_file_path)








