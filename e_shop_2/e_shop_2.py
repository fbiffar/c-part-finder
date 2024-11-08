# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
import requests
import json
from dotenv import load_dotenv  # For loading environment variables from a .env file
import os
import requests
from googlesearch import search
import csv



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
                    {"type": "text", "text": "Please identify any machine components in this image and their position in pixel coordinates. No rigorous explanation. List the components and their corresponding positions seperated by a plus sign and the component location pair seperated by commas."},
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

def print_response_and_store (response):
    # Extract and print only the content
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "No content available")
    print(f"Tile {idx + 1} Response: {content}")
    return [element.strip() for element in content.split(",") if element.strip()]
    
 


def search_bossard(component_name):
    query = f"Bossard {component_name} site: www.bossard.com"
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
def store_in_csv(file_path,part_id, component, url):
    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([part_id, component, url])

# Load and display the original image
image_path = "e_shop_2/conveyor_machine.png"
image = cv2.imread(image_path)
display_image(image, "Original Image")

# Divide the image into tiles
rows, cols = 2,2
tiles, tile_h, tile_w = divide_image_into_tiles(image, rows, cols)

# Overlay the grid on the image and display
image_with_grid = overlay_grid(image, rows, cols)
display_image(image_with_grid, "Image with Grid Overlay")

# Display each tile individually
for idx, tile in enumerate(tiles):
    display_image(tile, f"Tile {idx + 1}")

# Arrange and display all tiles in a single grid layout
tiles_grid = arrange_tiles_in_grid(tiles, rows, cols)
display_image(tiles_grid, "All Tiles Arranged in Grid")

# Encode each tile to base64 and send to OpenAI API
load_dotenv()

# Access environment variables
api_key = os.getenv("OPENAI_API_KEY")
api_endpoint = os.getenv("OPENAI_API_ENDPOINT")

all_tile_elements = []
for idx, tile in enumerate(tiles):
        encoded_tile = encode_to_base64(tile)
        response = send_to_openai_api(encoded_tile, api_endpoint, api_key)
        all_tile_elements.extend(print_response_and_store(response))
        break
# Print the combined list of all elements
print("All Elements from Tiles:", all_tile_elements)

availability = {}
# Initialize CSV file with header
csv_file_path = "machine_parts_live.csv"
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Part ID","Machine Part", "URL"])

for part_id, component in enumerate(all_tile_elements, start=1): 
    url = search_bossard(component)
    availability[component] = url
    print(f"Part ID: {part_id}, Component: {component} - URL: {url}")
    store_in_csv(csv_file_path, part_id, component, url) 
    break 

# Print the availability of each component