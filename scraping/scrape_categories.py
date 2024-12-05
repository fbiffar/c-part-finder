import os
import json
import re
from playwright.sync_api import sync_playwright

def sanitize_filename(name):
    # Replace spaces with underscores and remove any non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9]', '_', name)


def scrape_categories(page, base_url, url, categories, parent_category=None):
    # Navigate to the given URL
    page.goto(url)
    page.wait_for_load_state("networkidle")

    # Select all <a> elements with the class 'tile'
    locator = page.locator('a.tile')
    count = locator.count()
    print(f"Found {count} <a> elements with class 'tile'.")
    category_links = []
    for i in range(count):
        href = locator.nth(i).get_attribute('href')
        if href and not href.startswith('http'):
            href = base_url + href  # Prepend base URL if href is relative

        img_locator = locator.nth(i).locator('img.image--subcategory, img.image--category')
        category_type = None
        img_src = None
        if img_locator.count() > 0:
            img_src = img_locator.get_attribute('data-original') or img_locator.get_attribute('src')
            category_type = 'subcategory' if 'image--subcategory' in img_locator.get_attribute('class') else 'category'
            print(f"Element {i + 1} image URL: {img_src}")

        h3_locator = locator.nth(i).locator('h3.headingClasses')
        category_name = None
        if h3_locator.count() > 0:
            category_name = h3_locator.text_content().strip()
            print(f"Element {i + 1} H3 text content: {category_name}")

        if img_src and category_name:
            sanitized_name = sanitize_filename(category_name)
            img_data = page.evaluate(f"""
                async () => {{
                    const response = await fetch('{img_src}');
                    const buffer = await response.arrayBuffer();
                    return Array.from(new Uint8Array(buffer));
                }}
            """)
            img_bytes = bytes(img_data)
            img_path = f'./img/{sanitized_name}.jpeg'
            with open(img_path, 'wb') as img_file:
                img_file.write(img_bytes)
            print(f"Image for element {i + 1} saved as {img_path}.")

            category_entry = {
                "category_type": category_type,
                "category_name": category_name,
                "category_link": href,
                "category_img": img_path,
                "parent_category": parent_category
            }
            categories.append(category_entry)
            # Collect category links for later recursive scraping
            if category_type == 'category' and href:
                category_links.append((href, category_name))

    # After collecting all categories, recursively scrape subcategories
    for link, name in category_links:
        scrape_categories(page, base_url, link, categories, name)

def main():
    os.makedirs('./img', exist_ok=True)
    categories = []
    base_url = "https://www.bossard.com"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Start scraping from the main page
        scrape_categories(page, base_url, f"{base_url}/eshop/ch-de/", categories)

        browser.close()

    with open('categories.json', 'w', encoding='utf-8') as json_file:
        json.dump({"categories": categories}, json_file, ensure_ascii=False, indent=4)
    print("Categories saved to categories.json")

if __name__ == "__main__":
    main()
