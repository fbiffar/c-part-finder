import json

def extract_category_subcategory_pairs(json_path):
    """Extract category-subcategory pairs from the JSON structure."""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    pairs = []
    for category in data.get("categories", []):
        category_name = category.get("category_name")
        for subcategory in category.get("subcategories", []):
            subcategory_name = subcategory.get("category_name")
            pairs.append((category_name, subcategory_name))
    
    return pairs


def extract_category_details(json_path, category_name, subcategory_name):
    """
    Extract details for a given category and subcategory from the JSON structure.

    Args:
    - json_path: Path to the JSON file.
    - category_name: The name of the category to search for.
    - subcategory_name: The name of the subcategory to search for.

    Returns:
    - A tuple (category_name, subcategory_name, link, image_path) or None if not found.
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Traverse the JSON structure
    for category in data.get("categories", []):
        if category.get("category_name") == category_name:
            for subcategory in category.get("subcategories", []):
                if subcategory.get("category_name") == subcategory_name:
                    link = subcategory.get("category_link")
                    image_path = subcategory.get("category_img")
                    return (category_name, subcategory_name, link, image_path)

    # Return None if no match is found
    return None
