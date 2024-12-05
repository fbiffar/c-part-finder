import json

def restructure_categories(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories = data['categories']
    restructured = {}

    # First, create a dictionary for parent categories
    for category in categories:
        if category['category_type'] == 'category':
            restructured[category['category_name']] = {
                "category_type": category['category_type'],
                "category_name": category['category_name'],
                "category_link": category['category_link'],
                "category_img": category['category_img'],
                "parent_category": category['parent_category'],
                "subcategories": []
            }

    # Then, add subcategories to their respective parent categories
    for category in categories:
        if category['category_type'] == 'subcategory':
            parent_name = category['parent_category']
            if parent_name in restructured:
                restructured[parent_name]['subcategories'].append({
                    "category_type": category['category_type'],
                    "category_name": category['category_name'],
                    "category_link": category['category_link'],
                    "category_img": category['category_img'],
                    "parent_category": category['parent_category']
                })

    # Convert the restructured dictionary back to a list
    restructured_list = list(restructured.values())

    # Write the restructured data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"categories": restructured_list}, f, ensure_ascii=False, indent=4)

# Example usage
restructure_categories('./categories.json', 'restructured_categories.json')