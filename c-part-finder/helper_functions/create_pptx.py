from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from PIL import Image
import numpy as np
import io
from helper_functions.json_handler import *

from pptx.oxml.ns import qn
from pptx.oxml import parse_xml

def add_hyperlink(run, url):
    """
    Add a hyperlink to a run of text in a PowerPoint slide.

    Args:
    - run: The text run to which the hyperlink will be added.
    - url: The URL for the hyperlink.
    """
    # Create a relationship for the hyperlink
    rId = run.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)

    # Add the hyperlink XML element
    rPr = run._r.get_or_add_rPr()  # Get or create the run properties
    hyperlink_xml = f'<a:hlinkClick xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" r:id="{rId}"/>'
    rPr.append(parse_xml(hyperlink_xml))


def create_pptx_with_annotations(img, annotations, output_path="annotated_presentation.pptx"):
    """
    Generate a PowerPoint slide with the given image and annotations.

    Parameters:
    - img: PIL.Image instance to be inserted.
    - annotations: List of tuples containing annotation details [(coords, category_name, subcategory_name, url), ...].
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
    slide.shapes.add_picture(img_bytes, image_left, image_top, width=image_width_on_slide, height=image_height_on_slide)

    # Calculate scaling factors for annotation placement
    scale_x = image_width_on_slide / img_width
    scale_y = image_height_on_slide / img_height

    # Free space dimensions for labels and images
    free_space_left = image_left - Inches(2.0)  # Margin for labels on the left
    free_space_right = image_left + image_width_on_slide + Inches(0.2)  # Margin for labels on the right
    free_space_width = Inches(2.0)  # Fixed width for labels and images

    # Initial top position for labels and images
    current_label_top_left = Inches(0.5)  # Start from the top of the slide for left side
    current_label_top_right = Inches(0.5)  # Start from the top of the slide for right side

    # Add annotations
    for idx, (coords, category_name, subcategory_name, url) in enumerate(annotations):
        # Unpack coordinates
        left, top, right, bottom = coords
        part_info = extract_category_details(json_path, category_name, subcategory_name)
        path_image = None
        if part_info:
            path_image = part_info[3]

        # Scale coordinates to match PowerPoint dimensions
        left_scaled = left * scale_x + image_left
        top_scaled = top * scale_y + image_top
        right_scaled = right * scale_x + image_left
        bottom_scaled = bottom * scale_y + image_top

        # Calculate center of the marked region for the arrow
        marked_center_x = (left_scaled + right_scaled) / 2
        marked_center_y = (top_scaled + bottom_scaled) / 2

        # Determine label position (left or right side)
        if idx % 2 == 0:  # Even indices go to the right side
            label_left = free_space_right
            current_label_top = current_label_top_right
            current_label_top_right += Inches(1.0)  # Increment for the next right-side annotation
        else:  # Odd indices go to the left side
            label_left = free_space_left
            current_label_top = current_label_top_left
            current_label_top_left += Inches(1.0)  # Increment for the next left-side annotation

        label_height = Inches(0.5)  # Fixed height for labels
        image_spacing = Inches(0.2)  # Space between label and image

        # Add label textbox
        label = slide.shapes.add_textbox(label_left, current_label_top, free_space_width, label_height)
        text_frame = label.text_frame
        p = text_frame.add_paragraph()
        run = p.add_run()
        run.text = subcategory_name
        run.font.size = Pt(12)  # Use Pt for font size
        run.font.bold = True

        # Add hyperlink to the label
        # if url:
        #     add_hyperlink(run, url)

        # Add subcategory image below the label
        image_bottom_y = current_label_top + label_height + image_spacing
        if path_image:
            img_part = Image.open(path_image)
            img_bytes_part = io.BytesIO()
            img_part.save(img_bytes_part, format="PNG")
            img_bytes_part.seek(0)
            
            # Calculate the size for the small image
            part_image_width = Inches(1.5)
            part_aspect_ratio = img_part.height / img_part.width
            part_image_height = part_image_width * part_aspect_ratio

            slide.shapes.add_picture(
                img_bytes_part,
                label_left,
                image_bottom_y,  # Position below the label
                width=part_image_width,
                height=part_image_height
            )

            # Update the bottom position for the next annotation
            current_label_top = image_bottom_y + part_image_height + Inches(0.5)
        else:
            # If no image, increment the current label top just for the label and spacing
            current_label_top += label_height + Inches(0.5)

        # Add connecting arrow/line from the marked region to the label
        arrow = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT,
            marked_center_x,
            marked_center_y,
            label_left + free_space_width / 2,  # Center of the label box
            current_label_top - Inches(1.0)  # Adjust to point at the label
        )
        arrow.line.color.rgb = RGBColor(0, 0, 255)  # Blue arrow

    # Save the PowerPoint presentation
    prs.save(output_path)
    print(f"PowerPoint saved to {output_path}")


# annotation = [((112, 215, 187, 301), 'Flüssigkeitsmanagement', 'Hydraulische Komponenten', 'https://www.bossard.com/eshop/ch-de/produkte/funktionselemente/fluessigkeitsmanagement/hydraulische-komponenten/c/03.300.200/'), ((196, 38, 251, 86), 'Norm- und Standard Verbindungselemente', 'Schrauben', 'https://www.bossard.com/eshop/ch-de/produkte/verbindungstechnik/norm-und-standard-verbindungselemente/schrauben/c/01.100.100/')]
# img_path = "c-part-finder/images/conveyor_machine.png"
# img = Image.open(img_path)
# create_pptx_with_annotations(img, annotation, output_path="annotated_presentation.pptx")