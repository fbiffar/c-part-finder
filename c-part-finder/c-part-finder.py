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
from sklearn.metrics.pairwise import cosine_similarity
import os
from helper_functions.create_pptx import *
from helper_functions import json_handler
from helper_functions import df_handler
import io



load_dotenv()
# Initialize OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=api_key)

# Pydantic model for OpenAI API response
class MachinePartIdentifier(BaseModel):
    category_name: str
    subcategory_name: str
    certainty: float

class ManualPartIdentifier(BaseModel):
    unique_id: int
    name: str

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
    img_height, img_width = annotated.shape[:2]  # Get the image dimensions
    left, top, right, bottom = roi_coords
    green = (0, 255, 0)  # Green color for the rectangle
    black = (0, 0, 0)    # Black color for the text

    # Draw the green rectangle
    cv2.rectangle(annotated, (left, top), (right, bottom), green, 2)  # Green rectangle

    # Add label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(part_name, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    # Calculate text position
    text_x = left
    text_y = top - 10  # Position text above the rectangle

    # Ensure text stays within the image width
    if text_x + text_width > img_width:
        text_x = img_width - text_width - 5  # Shift text to the left if it overflows

    # Ensure text stays within the image height
    if text_y - text_height - 5 < 0:
        text_y = top + text_height + 10  # Move text below the rectangle if it overflows

    # Draw the green background for the text
    cv2.rectangle(annotated, (text_x, text_y - text_height - 5),
                  (text_x + text_width, text_y + 5), green, -1)  # Green background

    # Add the black text
    cv2.putText(annotated, part_name, (text_x, text_y), font, font_scale, black, thickness)

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
    
def convert_to_csv(annotate):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df = pd.DataFrame(annotate, columns=["ROI Coordinates", "Part ID", "Category", "URL"])

    return df.to_csv().encode('utf-8')


# Function to perform semantic search
def semantic_search(query_embedding, embeddings, top_n=3):
    # Compute cosine similarity between the query and all embeddings
    similarities = cosine_similarity([query_embedding], embeddings)
    # Get the indices of the top_n most similar embeddings
    top_indices = np.argsort(similarities[0])[-top_n:][::-1]
    print(f"Top indices: {top_indices}")
    top_similarities = similarities[0][top_indices]           
    return top_indices, top_similarities

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_part_description(encoded_image,api_key, context_partnames=""):

    # Step 1: Generate a description using OpenAI's language model
    prompt = (
        """Analyze the provided image of a machine part and generate a two-sentence description "
        "of the visible parts in the marked area and what their function is and what they could be in the context of the surrounding machine or product. """
    )
    
    client = OpenAI(api_key=api_key)
    # Use the OpenAI API to generate the description
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": prompt
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
        ]
    )
    print(f"Description: {response}")
    description = response.choices[0].message.content

    # Step 2: Embed the description
    # Assuming you have a function or model to embed text, e.g., using OpenAI's embedding API
    embedding_response = get_embedding(description)
    
    # Extract the embedding
    #embedded_description = embedding_response['data'][0]['embedding']

    return embedding_response

def send_to_label_openai_api(encoded_image, api_endpoint, api_key, pairs, label):
    """
    Sends an image and label to the OpenAI API and retrieves the best matching industrial component.

    Args:
        encoded_image (str): Base64 encoded image with the marked region.
        api_endpoint (str): API endpoint URL.
        api_key (str): API key for authentication.
        pairs (list): List of dictionaries containing 'id' and 'name' pairs.
        label (str): Label name provided as the primary indicator for matching.

    Returns:
        ManualPartIdentifier: Object containing the matched 'unique_id', 'name'.
    """
    client = OpenAI(api_key=api_key)

    # Define the messages for the API request
    messages = [
        {
            "role": "system",
            "content": f"""
            Du bist ein Experte für die Identifikation industrieller Komponenten. Deine Aufgabe ist es, den bereitgestellten Label-Namen mit der höchsten Priorität zu verwenden, um das am besten passende Element aus der bereitgestellten Liste zu bestimmen. Wenn die Übereinstimmung mit dem Label nicht eindeutig ist, kannst du das bereitgestellte Bild zur Unterstützung verwenden.

            Liste der Elemente:
            {pairs}
            Label-Name: {label}

            Schritte:
            1. Vergleiche den bereitgestellten Label-Namen mit den Namen in der Liste.
            2. Gib den eintrag zurück der am besten passt.
            3. Wenn die Übereinstimmung nicht eindeutig ist, kannst du den rot markierten Bereich im Bild analysieren.
            4. Identifiziere die Komponente oder das Bauteil, das im markierten Bereich zu sehen ist, und bestimme das Element (ID, Name), das am besten passt.
            5. Gib das passende Element sowie dein Vertrauen in die Übereinstimmung als Punktzahl zurück.

            Ausgabeformat:
            - Passendes Element: ID und Name des besten Treffers.
            
            Kontext:
            Nutze den bereitgestellten Label-Namen und, falls erforderlich, den rot markierten Bereich im Bild sowie den Kontext der Maschine oder des Produkts, um eine möglichst präzise Identifikation vorzunehmen.
            """
        },
        {
            "role": "user",
            "content": f"data:image/png;base64,{encoded_image}"
        }
    ]

    # Send the request to the OpenAI API
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=ManualPartIdentifier,
    )

    return response

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
    elif identification_mode == "embedding":
        unique_id = category_name
        part_info = df_handler.extract_category_details(csv_filepath = "context/unique_id_embeddings.csv",unique_id=unique_id)
        print(f"===================\nPart Link: {part_info['category_link']}")
        url = part_info["category_link"]
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
            st.image(roi)

            # Step 3: Choose Identification Mode
            st.write("### Identify Part")
            identification_mode = None
            part_name = None
            certainty = None

            col1, col2 = st.columns(2)
            print(f"ROI Size: {roi.shape}")
            encoded_roi = encode_image_to_base64(roi)
            # Auto Identify
            with col1:
                if st.button("Auto Identify Part"):
                    use_embedding = True
                    if use_embedding:
                        category_context = ""
                        with open("./context/context_categories.txt", 'r', encoding='utf-8') as file:
                            category_context = file.read()
                        category_context="Schrauben, Scharniere, Dichtscheiben, Muttern, Rollen, Kabelbinder, Kugelbolzen, Handräder"
                        embedding = get_part_description(encoded_roi, api_key, context_partnames=category_context)
                        df = pd.read_csv('context/unique_id_embeddings.csv')
                        unique_ids = df['unique_id'].tolist()
                        names = df['name'].tolist()
                        urls = df["category_link"].tolist()
                        imgs = df["category_img"].tolist()
                        embeddings = df['embedding'].apply(eval).tolist()  # Assuming embeddings are stored as strings
                        # Find the 3 most relevant embeddings
                        top_indices, top_similarities = semantic_search(embedding, embeddings)
                        top_unique_ids = [unique_ids[i] for i in top_indices]
                        top_unique_names = [names[i] for i in top_indices]
                        print("Top 3 most relevant unique IDs:", top_unique_ids)
                        print("Top 3 most relevant names:",top_unique_names)
                        print("Top 3 similarity values:", top_similarities)
                        part_category_name = top_unique_ids[0]                       
                        part_subcategory_name = names[part_category_name]
                        certainty = top_similarities[0]
                        identification_mode = "embedding"
                    else:

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
                        df = pd.read_csv('context/unique_id_embeddings.csv')
                        #pairs = df['unique_id','name'].tolist()
                        pairs = df[['unique_id', 'name']].values.tolist()
                        response = send_to_label_openai_api(encoded_roi, api_endpoint, api_key, pairs,part_subcategory_name)
                        certainty = 1.0  # Assume 100% certainty for manual entry
                        identification_mode = "embedding"
                        if response and response.choices:
                            parsed_data = response.choices[0].message.parsed
                            print(f"Manual Part Identifier: {parsed_data}")
                            part_category_name = parsed_data.unique_id                   
                            part_subcategory_name = parsed_data.name

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
                st.image(preview_image, caption="Preview of Annotated Image")


        # Step 5: Display Final Image
        st.write("### Current Image")
        st.image(st.session_state.final_output, caption="Final Output Image")

        # Step 6: Display Annotations List
        if st.session_state.annotations:
            display_annotations(st.session_state.annotations)

        # # Step 7: Save Results
        # if st.button("Save Annotated Image with Links"):
           
           
        col1, col2 = st.columns(2)
        with col1: 
            prs = create_pptx_with_annotations(img, st.session_state.annotate, output_path="annotated_presentation.pptx")
            # save the output into binary form
            binary_output = io.BytesIO()
            prs.save(binary_output) 
            st.download_button(label = 'Download PowerPoint',
                            data = binary_output.getvalue(),
                            file_name = 'machine_part.pptx')
        with col2:
            # Use Streamlit's download button
            st.download_button(
            label="Download data as CSV",
            data=convert_to_csv(st.session_state.annotate),
            file_name='machine_part.csv',
            mime='text/csv',
)


if __name__ == "__main__":
    main()
