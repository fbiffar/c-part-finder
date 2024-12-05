from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from PIL import Image
import numpy as np
import io
import json_handler

from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

def add_hyperlink(run, url):
    """
    Add a hyperlink to a run of text in a PowerPoint slide.
    """
    r_id = run.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink_xml = (
        f'<a:hlinkClick xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" r:id="{r_id}"/>'
    )
    run._r.get_or_add_hlinkClick()._element.append(parse_xml(hyperlink_xml))

def create_pptx_with_annotations(img, annotations, output_path="annotated_presentation.pptx"):
    """
    Generate a PowerPoint slide with the given image and annotations.
    
    Parameters:
    - image_path: Path to the image to be inserted.
    - annotations: List of tuples containing annotation details [(coords, part_name), ...].
    - output_path: Path where the generated PowerPoint file will be saved.
    """
    json_path = "c-part-finder/structured_categories/restructured_categories.json"
    
    # Convert the PIL Image to a file-like object (BytesIO)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)  # Reset the file pointer to the beginning

    img_width, img_height = img.size

    # Create PowerPoint presentation
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank slide layout

    # Get slide dimensions
    slide_width = prs.slide_width
    slide_height = prs.slide_height

    # Calculate the desired image width (50% of slide width) and maintain aspect ratio
    image_width_on_slide = slide_width * 0.5
    image_aspect_ratio = img_height / img_width
    image_height_on_slide = image_width_on_slide * image_aspect_ratio

    # Center the image on the slide
    image_left = (slide_width - image_width_on_slide) / 2  # Center horizontally
    image_top = (slide_height - image_height_on_slide) / 2  # Center vertically
    slide_image = slide.shapes.add_picture(img_bytes, image_left, image_top, width=image_width_on_slide, height=image_height_on_slide)

    # Calculate scaling factors for annotation placement
    scale_x = image_width_on_slide / img_width
    scale_y = image_height_on_slide / img_height

    # Calculate free space for labels (outside the image)
    free_space_left = 0 if image_left > slide_width / 2 else image_left + image_width_on_slide
    free_space_width = abs(slide_width - image_width_on_slide) / 2

    # Center labels vertically in the free space
    label_top_start = (slide_height - len(annotations) * Inches(0.7)) / 2  # Dynamically position labels

    # Add annotations
    for idx, (coords, category_name, subcategory_name, url) in enumerate(annotations):
        # Unpack coordinates
        left, top, right, bottom = coords
        # part_info = json_handler.extract_category_details(json_path, category_name, subcategory_name)
        # path_image = part_info[3]
        # img_part = Image.open(path_image)

        # Scale coordinates to match PowerPoint dimensions
        left_scaled = left * scale_x + image_left
        top_scaled = top * scale_y + image_top
        right_scaled = right * scale_x + image_left
        bottom_scaled = bottom * scale_y + image_top

        # Draw the box around the part
        shape = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, left_scaled, top_scaled,
            right_scaled - left_scaled, bottom_scaled - top_scaled
        )
        shape.line.color.rgb = RGBColor(255, 0, 0)  # Red border
        shape.fill.background()  # Transparent fill

        # Position label dynamically in free space
        label_left = free_space_left + Inches(0.2)  # Add margin
        label_top = label_top_start + idx * Inches(0.7)  # Stack labels vertically

        # Add an arrow from the rectangle to the label
        arrow = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, right_scaled, (top_scaled + bottom_scaled) / 2,
            label_left, label_top + Inches(0.25)
        )
        arrow.line.color.rgb = RGBColor(0, 0, 255)  # Blue arrow

        # Add the label with the part name
        label = slide.shapes.add_textbox(label_left, label_top, free_space_width - Inches(0.4), Inches(0.5))
        text_frame = label.text_frame
        text_frame.text = subcategory_name
        text_frame.paragraphs[0].font.size = Pt(12)  # Use Pt for font size
        text_frame.paragraphs[0].font.bold = True

        # Add hyperlink to the label

        # Add Image below the label



    # Save the PowerPoint presentation
    prs.save(output_path)
    print(f"PowerPoint saved to {output_path}")


