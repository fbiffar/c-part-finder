



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
from openai import OpenAI

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
    # Convert BGR to RGB if color image
    if len(img.shape) == 3:
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        display_img = img
        
    # Display image with streamlit
    st.subheader(title)
    st.image(display_img, use_column_width=True)

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

def annotate_tile(tile, part_name):
    """Add circle and text label to a tile."""
    annotated_tile = tile.copy()
    height, width = tile.shape[:2]
    center = (width // 2, height // 2)
    radius = min(width, height) // 4  # Circle radius as 1/4 of smallest dimension
    
    # Draw circle
    cv2.circle(annotated_tile, center, radius, (0, 255, 0), 2)  # Green circle
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(part_name, font, font_scale, thickness)[0]
    
    # Position text above the circle
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] - radius - 10
    
    # Add white background for text
    padding = 5
    cv2.rectangle(annotated_tile, 
                 (text_x - padding, text_y - text_size[1] - padding),
                 (text_x + text_size[0] + padding, text_y + padding),
                 (255, 255, 255),
                 -1)
    
    # Add text
    cv2.putText(annotated_tile, part_name, (text_x, text_y), 
                font, font_scale, (0, 0, 0), thickness)
    
    return annotated_tile

def process_and_display_results(image, tiles, responses):
    """Process tiles with annotations and create final display."""
    annotated_tiles = []
    rows = cols = int(len(tiles) ** 0.5)  # Assuming square grid
    
    for idx, tile in enumerate(tiles):
        part_name = responses[idx].choices[0].message.content
        annotated_tile = annotate_tile(tile, part_name)
        annotated_tiles.append(annotated_tile)
    
    # Stitch tiles back together
    tile_h, tile_w = tiles[0].shape[:2]
    final_image = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)
    
    for idx, tile in enumerate(annotated_tiles):
        i, j = divmod(idx, cols)
        final_image[i * tile_h:(i + 1) * tile_h, 
                   j * tile_w:(j + 1) * tile_w] = tile
    
    return final_image

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

def arrange_tiles_in_grid(tiles, rows, cols):
    # Convert all tiles to BGR first
    converted_tiles = []
    for tile in tiles:
        if tile.shape[2] == 4:  # If image has alpha channel
            converted_tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2BGR)
            converted_tiles.append(converted_tile)
        else:
            converted_tiles.append(tile)
    
    tile_h, tile_w = converted_tiles[0].shape[:2]
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)  # Always create 3-channel grid
    
    for idx, tile in enumerate(converted_tiles):
        i, j = divmod(idx, cols)
        grid[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w] = tile
    
    return grid

# Function to encode an image to base64
def encode_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# Function to send image data to OpenAI API
def send_to_openai_api(encoded_image, api_endpoint, api_key):
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a machine parts identification expert. When shown an image, identify only the single most prominent machine part visible. Respond with just the generic name of the part (e.g., 'bolt', 'washer', 'bearing'). If no machine parts are visible, respond with 'No machine parts found'."
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    return response

def print_response_and_store(response, idx):
    # Extract and print only the content
    content = response.choices[0].message.content
    print(f"Tile {idx + 1} Response: {content}")
    return [[element.strip(), idx + 1] for element in content.split(",") if element.strip()]

 


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







# Modify your existing code to include annotation:
image, rows, cols = get_image_from_finder()
display_image(image, "Original Image")
tiles, tile_h, tile_w = divide_image_into_tiles(image, rows, cols)

load_dotenv()

# Access environment variables
api_key = os.getenv("OPENAI_API_KEY")
api_endpoint = os.getenv("OPENAI_API_ENDPOINT")

all_tile_elements = []
annotated_tiles = []  # New list to store annotated tiles

for idx, tile in enumerate(tiles):
    encoded_tile = encode_to_base64(tile)
    response = send_to_openai_api(encoded_tile, api_endpoint, api_key)
    elements = print_response_and_store(response, idx)
    all_tile_elements.extend(elements)
    
    # Annotate the tile with the identified part name
    part_name = elements[0][0] if elements else "No part found"
    annotated_tile = annotate_tile(tile, part_name)
    annotated_tiles.append(annotated_tile)

# Stitch annotated tiles back together
final_image = arrange_tiles_in_grid(annotated_tiles, rows, cols)

# Display the annotated image
display_image(final_image, "Analyzed Machine Parts")

# Continue with your existing CSV processing
print("All Elements from Tiles:", all_tile_elements)

availability = {}
csv_file_path = "machine_parts_live.csv"
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Part ID","Machine Part", "URL", "Tile ID"])

for part_id, component in enumerate(all_tile_elements, start=1): 
    component_name, tile_id = component
    url = component_name    #search_bossard(component_name)
    availability[component_name] = url
    print(f"Part ID: {part_id}, Component: {component_name} - URL: {url}")
    store_in_csv(csv_file_path, part_id, component_name, url, tile_id) 

dialogue_return_csv(csv_file_path)








