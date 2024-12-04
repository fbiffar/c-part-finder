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
from streamlit_drawable_canvas import st_canvas

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
        st.image(image, caption="Uploaded Image", use_container_width=True)

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
    st.image(display_img, use_container_width=True)

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
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a machine parts identification expert. When shown an image, identify only the single most prominent machine part visible. Respond with the name of the part (e.g., 'bolt', 'washer', 'bearing', 'hinge', 'e-stop', etc.). If no machine parts are visible, respond with 'No part'."
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
        max_tokens=800
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
        result = next(search(query, num=1, lang='de'), None)
        if result:
            return result
        else:
            return "Not available"
    except Exception as e:
        print(f"Error searching for '{component_name}': {e}")
        return "Not available"
    
# Function to store the component and its URL in the CSV file immediately
def store_in_csv(file_path, part_id, component, url, tile_id):
    """Store component information in CSV if valid part and URL found."""
    if component != 'No Part' and url != "Not available":
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
def get_image_from_finder():
    st.title("Machine Part Identifier")

    # File uploader for the image
    uploaded_file = st.file_uploader("Upload an image of machine", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Open the uploaded image
        image_pl = Image.open(uploaded_file)
        image = np.array(image_pl)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        return image
    
    return None

def annotate_part(image, roi_coords, part_name):
    """Add annotation box and label to the image."""
    annotated = image.copy()
    left, top, right, bottom = roi_coords
    
    # Draw rectangle in green
    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Add text with background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(part_name, font, font_scale, thickness)[0]
    
    # Position text above the box
    text_x = left
    text_y = top - 10 if top - 10 > text_size[1] else top + text_size[1]
    
    # Add white background for text
    cv2.rectangle(annotated, 
                 (text_x, text_y - text_size[1] - 5),
                 (text_x + text_size[0], text_y + 5),
                 (255, 255, 255),
                 -1)
    
    # Add text in black
    cv2.putText(annotated, part_name, (text_x, text_y), 
                font, font_scale, (0, 0, 0), thickness)
    
    return annotated

def select_roi(image):
    st.write("Draw a rectangle around the part you want to identify")
    
    with st.container():
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Calculate scaling factor to fit screen width
        max_display_width = 1400
        scale_factor = min(1.0, max_display_width / img_width)
        
        # Calculate canvas dimensions
        canvas_width = int(img_width * scale_factor)
        canvas_height = int(img_height * scale_factor)
        
        # Create canvas for drawing
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Semi-transparent orange
            stroke_width=2,
            stroke_color="#ff0000",  # Red outline for drawing
            background_image=Image.fromarray(image),
            drawing_mode="rect",
            key="canvas",
            width=canvas_width,
            height=canvas_height,
            display_toolbar=True,
        )
    
    if (canvas_result.json_data is not None and 
        "objects" in canvas_result.json_data and 
        len(canvas_result.json_data["objects"]) > 0):
        
        rect = canvas_result.json_data["objects"][-1]
        
        # Calculate scale factors
        scale_x = img_width / canvas_width
        scale_y = img_height / canvas_height
        
        # Extract coordinates and scale them
        left = int(rect["left"] * scale_x)
        top = int(rect["top"] * scale_y)
        width = int(rect["width"] * scale_x)
        height = int(rect["height"] * scale_y)
        
        # Ensure coordinates are within bounds
        right = min(left + width, img_width)
        bottom = min(top + height, img_height)
        left = max(0, left)
        top = max(0, top)
        
        # Extract ROI
        roi = image[top:bottom, left:right]
        
        if roi.size > 0:
            st.write("Selected Region (Original Size):")
            st.image(roi, use_container_width=False)
            # Return the exact coordinates that were used for the ROI
            return roi, (int(left), int(top), int(right), int(bottom))
    
    return None, None

def save_annotated_image_with_links(image, annotations, output_path="annotated_output.html"):
    """Save image with clickable regions and links."""
    from PIL import Image
    import io
    
    # Convert BGR to RGB for correct color display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    
    # Save image to base64 for HTML embedding
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Create HTML with image map
    html = f"""
    <html>
    <body>
    <img src="data:image/png;base64,{img_str}" usemap="#partmap">
    <map name="partmap">
    """
    
    # Add clickable regions
    for coords, part_name, url in annotations:
        left, top, right, bottom = coords
        html += f'<area shape="rect" coords="{left},{top},{right},{bottom}" href="{url}" title="{part_name}">\n'
    
    html += """
    </map>
    <h2>Parts List:</h2>
    <ul>
    """
    
    # Add list of parts and links
    for coords, part_name, url in annotations:
        if url != "Not available":
            html += f'<li>{part_name}: <a href="{url}">{url}</a></li>\n'
        else:
            html += f'<li>{part_name}: No link available</li>\n'
    
    html += """
    </ul>
    </body>
    </html>
    """
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path

def main():
    st.set_page_config(layout="wide")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    api_endpoint = os.getenv("OPENAI_API_ENDPOINT")

    # Initialize session state for annotations if it doesn't exist
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
        st.session_state.final_output = None

    # Get and display image
    image = get_image_from_finder()
    
    if image is not None:
        # Initialize final output if needed
        if st.session_state.final_output is None:
            st.session_state.final_output = image.copy()
        
        # Select ROI
        roi, roi_coords = select_roi(image)
        
        if roi is not None and st.button("Identify Part"):
            # Encode and send ROI to OpenAI
            encoded_roi = encode_to_base64(roi)
            response = send_to_openai_api(encoded_roi, api_endpoint, api_key)
            
            # Get part name from response
            part_name = response.choices[0].message.content
            
            # Search for part
            url = search_bossard(part_name)
            
            # Store annotation
            st.session_state.annotations.append((roi_coords, part_name, url))
            
            # Update final output image
            st.session_state.final_output = annotate_part(
                st.session_state.final_output, 
                roi_coords, 
                part_name
            )
            
            # Display results
            st.write(f"Identified Part: {part_name}")
            if url != "Not available":
                st.markdown(f"[View Part on Bossard]({url})")
            else:
                st.write("No product link available")
        
        # Display final output image
        if st.session_state.final_output is not None:
            st.write("Final Output Image:")
            st.image(st.session_state.final_output, use_container_width=True)
        
        # Display list of identified parts and links
        if st.session_state.annotations:
            st.write("### Identified Parts:")
            for coords, part_name, url in st.session_state.annotations:
                if url != "Not available":
                    st.markdown(f"- {part_name}: [{url}]({url})")
                else:
                    st.markdown(f"- {part_name}: No link available")
        
        # Add button to save annotated image with links
        if st.button("Save Annotated Image with Links"):
            output_path = save_annotated_image_with_links(
                st.session_state.final_output,
                st.session_state.annotations
            )
            st.success(f"Saved annotated image with links to {output_path}")

if __name__ == "__main__":
    main()







