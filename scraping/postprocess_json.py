import json

def add_subcategories(category, categories_dict):
    """Recursively add subcategories to a category."""
    subcategories = [
        subcat for subcat in categories_dict.values()
        if subcat['parent_category'] == category['category_name']
    ]
    for subcat in subcategories:
        subcat['subcategories'] = add_subcategories(subcat, categories_dict)
    return subcategories

def restructure_categories(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories = data['categories']
    categories_dict = {cat['category_name']: cat for cat in categories}

    # Initialize subcategories list for each category
    for category in categories:
        category['subcategories'] = []

    # Build the hierarchy
    restructured = []
    for category in categories:
        if category['parent_category'] is None:
            category['subcategories'] = add_subcategories(category, categories_dict)
            restructured.append(category)

    # Write the restructured data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"categories": restructured}, f, ensure_ascii=False, indent=4)

# Example usage
restructure_categories('categories.json', 'restructured_categories.json')