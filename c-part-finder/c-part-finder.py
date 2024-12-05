# Run with: streamlit run e_shop_2.py

# Import necessary libraries
import cv2
import numpy as np
import base64
import requests
import csv
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from googlesearch import search
from pydantic import BaseModel
from openai import OpenAI
import os
from helper_functions.create_pptx import *
from helper_functions import json_handler

# Pydantic model for OpenAI API response
class MachinePartIdentifier(BaseModel):
    category_name: str
    subcategory_name: str
    certainty: float

# =====================
# Utility Functions
# =====================

def load_image():
    """Upload an image using Streamlit."""
    st.title("Machine Part Identifier")
    uploaded_file = st.file_uploader("Upload an image of machine", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        return np.array(Image.open(uploaded_file)), Image.open(uploaded_file)
    return None, None

def encode_image_to_base64(image):
    """Convert an image to base64 encoding."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def mark_roi_with_border(image, roi_coords):
    """Draw a red border around the selected ROI."""
    annotated_image = image.copy()
    left, top, right, bottom = roi_coords
    cv2.rectangle(annotated_image, (left, top), (right, bottom), (0, 0, 255), 2)  # Red border
    return annotated_image

def select_roi(image):
    """Allow users to select a region of interest (ROI) on the image."""
    st.write("Draw a rectangle around the part you want to identify")
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Adjust image size for canvas display
    max_display_width = 1400
    scale_factor = min(1.0, max_display_width / img_width)
    canvas_width = int(img_width * scale_factor)
    canvas_height = int(img_height * scale_factor)
    
    # Create drawable canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Transparent orange
        stroke_width=2,
        stroke_color="#ff0000",  # Red outline
        background_image=Image.fromarray(image),
        drawing_mode="rect",
        key="canvas",
        width=canvas_width,
        height=canvas_height,
        display_toolbar=True,
    )

    # Extract ROI coordinates if a rectangle is drawn
    if (canvas_result.json_data is not None and 
        "objects" in canvas_result.json_data and 
        len(canvas_result.json_data["objects"]) > 0):
        
        rect = canvas_result.json_data["objects"][-1]
        scale_x = img_width / canvas_width
        scale_y = img_height / canvas_height
        
        # Scale coordinates back to original image size
        left = int(rect["left"] * scale_x)
        top = int(rect["top"] * scale_y)
        width = int(rect["width"] * scale_x)
        height = int(rect["height"] * scale_y)
        
        # Calculate ROI bounds
        right = min(left + width, img_width)
        bottom = min(top + height, img_height)
        left = max(0, left)
        top = max(0, top)
        
        # Extract ROI and mark it with a border
        roi = image[top:bottom, left:right]
        image_with_roi = mark_roi_with_border(image, (left, top, right, bottom))
        
        return roi, (left, top, right, bottom)

    return None, None


def annotate_part(image, roi_coords, part_name):
    """Annotate the image with a green rectangle and part name."""
    annotated = image.copy()
    left, top, right, bottom = roi_coords
    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangle
    
    # Add label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(part_name, font, font_scale, thickness)[0]
    text_x = left
    text_y = max(top - 10, text_size[1] + 5)  # Prevent text from being outside the image
    cv2.rectangle(annotated, (text_x, text_y - text_size[1] - 5),
                  (text_x + text_size[0], text_y + 5), (255, 255, 255), -1)  # White background
    cv2.putText(annotated, part_name, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    return annotated

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
    
def save_annotations_to_csv(file_path, annotations):
    """Save annotations to a CSV file."""
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Part ID", "Component", "URL", "ROI Coordinates"])
        for idx, (roi_coords, part_name, url) in enumerate(annotations):
            writer.writerow([idx + 1, part_name, url, roi_coords])

def send_to_openai_api(encoded_image, api_endpoint, api_key, pairs):
    client = OpenAI(api_key=api_key)

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Du bist ein Experte für die Identifikation industrieller Komponenten. Deine Aufgabe ist es, den rot markierten Bereich in dem bereitgestellten Bild zu analysieren und das am besten passende Paar aus Kategorie und Unterkategorie aus der bereitgestellten Liste zu bestimmen.

                Liste der Kategorien und Unterkategorien:
                {pairs}

                Schritte:
                1. Analysiere sorgfältig den rot markierten Bereich im Bild.
                2. Identifiziere die Komponente oder das Bauteil, das im markierten Bereich zu sehen ist.
                3. Verwende die bereitgestellte Liste der Paare `(Kategorie, Unterkategorie)` als Referenz.
                4. Bestimme das Paar `(Kategorie, Unterkategorie)`, das am besten zu der identifizierten Komponente passt, höhere Priortät hat hier die subcategorie!!
                5. Gib das passende Paar sowie dein Vertrauen in die Übereinstimmung als Punktzahl zurück.

                Ausgabeformat:
                - Passendes Paar: Das exakte `(Kategorie, Unterkategorie)`-Paar, das am besten zu dem markierten Bereich passt.
                - Vertrauenspunktzahl: Ein Wert zwischen 0.0 und 1.0, der dein Vertrauen in die Identifikation angibt.

                Kontext:
                Nutze den rot markierten Bereich im Bild sowie den Kontext der Maschine oder des Produkts, um eine möglichst präzise Identifikation vorzunehmen. Die Kategorie liefert zusätzlichen Kontext zur Identifikation der Unterkategorie.
                """
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
        response_format=MachinePartIdentifier,
    )
    
    return response

def display_annotations(annotations):
    """Display the list of annotations with links."""
    st.write("### Identified Parts:")
    for _, (coords, part_name, url) in enumerate(annotations):
        if url != "Not available":
            st.markdown(f"- **{part_name}**: [View Part on Bossard]({url})")
        else:
            st.markdown(f"- **{part_name}**: No link available")


def handle_identification(category_name, subcategory_name, json_path, certainty, roi_coords, identification_mode):
    """
    Handle the shared logic for both auto and manual identification.
    """
    # Search for the URL and Image Path
    if identification_mode == "auto":
        part_info = json_handler.extract_category_details(json_path, category_name, subcategory_name)
        url = part_info[2]
    else: 
        url = search_bossard(subcategory_name)
    # Generate preview with annotation
    preview_image = annotate_part(st.session_state.final_output.copy(), roi_coords, f"{subcategory_name}")
    st.session_state.pending_annotation = (preview_image, roi_coords, category_name, subcategory_name, url)

    # Display preview
    st.write("### Annotation Preview")

    # Display identified part and link
    st.write(f"Identified Part: {category_name}, {subcategory_name}")
    st.write(f"Certainty: {certainty:.1%}")
    if url != "Not available":
        st.markdown(f"[View Part on Bossard]({url})")
    else:
        st.write("No product link available")

# =====================
# Main Workflow
# =====================

def main():
    st.set_page_config(layout="wide")
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    api_endpoint = os.getenv("OPENAI_API_ENDPOINT")

    # Initialize session state
    if "annotations" not in st.session_state:
        st.session_state.annotations = []
        st.session_state.final_output = None
        st.session_state.pending_annotation = None  # Hold pending annotation
        st.session_state.display_preview = False  # Control the display of the preview image
        st.session_state.annotate = []
    # Example Usage
    json_path = "c-part-finder/structured_categories/restructured_categories.json"
    pairs = json_handler.extract_category_subcategory_pairs(json_path)
    # Step 1: Upload image
    image, img = load_image()

    if image is not None:
        # Initialize final output
        if st.session_state.final_output is None:
            st.session_state.final_output = image.copy()

        # Step 2: Select ROI
        roi, roi_coords = select_roi(image)
        if roi is not None:
            st.write("Selected Region (Original Size):")
            st.image(roi, use_container_width=False)

            # Step 3: Choose Identification Mode
            st.write("### Identify Part")
            identification_mode = None
            part_name = None
            certainty = None

            col1, col2 = st.columns(2)

            # Auto Identify
            with col1:
                if st.button("Auto Identify Part"):
                    encoded_roi = encode_image_to_base64(roi)
                    response = send_to_openai_api(encoded_roi, api_endpoint, api_key, pairs)

                    if response and response.choices:
                        parsed_data = response.choices[0].message.parsed
                        part_category_name = parsed_data.category_name
                        part_subcategory_name = parsed_data.subcategory_name
                        certainty = parsed_data.certainty
                        identification_mode = "auto"

            # Manual Identify
            with col2:
                
                manual_part_name = st.text_input("Enter part name manually:")
                if st.button("Manually Identify"):
                    if manual_part_name:
                        part_subcategory_name = manual_part_name
                        part_category_name = "Manual Entry"
                        certainty = 1.0  # Assume 100% certainty for manual entry
                        identification_mode = "manual"
                    else:
                        st.error("Please enter a part name.")

            # Handle Identification Outside Columns
            if identification_mode and part_category_name and part_subcategory_name and certainty is not None:
                handle_identification(part_category_name,part_subcategory_name,json_path, certainty, roi_coords,identification_mode)
                st.session_state.display_preview = True


        
        # Step 4: Confirm or Reject Annotation
        if st.session_state.pending_annotation:
            preview_image, roi_coords, part_category_name, part_subcategory_name, url = st.session_state.pending_annotation

            # Actions Section: Buttons above the preview image
            st.write("### Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm Annotation"):
                    # Add to final annotations
                    st.session_state.final_output = preview_image
                    st.session_state.pending_annotation = None  # Clear pending annotation
                    st.session_state.display_preview = False
                    st.session_state.annotate.append((roi_coords,part_category_name, part_subcategory_name, url))
                    st.success("Annotation confirmed and added to the final image.")

            with col2:
                if st.button("Reject Annotation"):
                    st.session_state.pending_annotation = None  # Clear pending annotation
                    st.session_state.display_preview = False
                    st.info("Annotation rejected.")

            # Annotated Preview Image
            if st.session_state.display_preview:
                st.image(preview_image, caption="Preview of Annotated Image", use_container_width=True)


        # Step 5: Display Final Image
        st.write("### Current Image")
        st.image(st.session_state.final_output, caption="Final Output Image", use_container_width=True)

        # Step 6: Display Annotations List
        if st.session_state.annotations:
            display_annotations(st.session_state.annotations)

        # Step 7: Save Results
        if st.button("Save Annotated Image with Links"):
            print(st.session_state.annotate)
            create_pptx_with_annotations(img, st.session_state.annotate, output_path="annotated_presentation.pptx")
            output_path = "annotations.csv"
            save_annotations_to_csv(output_path, st.session_state.annotations)
            st.success(f"Annotations saved to {output_path}")

if __name__ == "__main__":
    main()
