import os
import json
import re
from playwright.sync_api import sync_playwright

def sanitize_filename(name):
    # Replace spaces with underscores and remove any non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9]', '_', name)


def main():
    # Ensure the image directory exists
    os.makedirs('./img', exist_ok=True)

    categories = []
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)  # Set headless=True for no UI
        context = browser.new_context()
        page = context.new_page()

        # Navigate to the website
        print("Navigating to the website...")
        page.goto("https://www.bossard.com/eshop/ch-de/")
        #page.goto("https://www.bossard.com/eshop/ch-de/produkte/funktionselemente/zugangs-und-verbindungsloesungen-und-bedienelemente/c/03.200/")

        # Wait for the "Zustimmen" button and click it
        print("Accepting cookies...")
        page.get_by_role("button", name="Zustimmen").wait_for(timeout=5000)
        page.get_by_role("button", name="Zustimmen").click()

        # Wait for the page to fully load
        print("Waiting for page to load...")
        page.wait_for_load_state("networkidle")  # Wait until no network activity

        # Find the "Einpresstechnik" link
        print("Locating 'Einpresstechnik' link...")
        einpresstechnik_locator = page.locator('a:has-text("Einpresstechnik") >> nth=0')

        # Wait for the element to be visible
        einpresstechnik_locator.wait_for(timeout=5000)

        # Print tag name
        tag_name = einpresstechnik_locator.evaluate("el => el.tagName")
        print(f"Tag Name: {tag_name}")


        # Print all attributes
        attributes = einpresstechnik_locator.evaluate("""
        el => {
            const attrs = {};
            for (const attr of el.attributes) {
                attrs[attr.name] = attr.value;
            }
            return attrs;
        }
        """)
        print("Attributes:", attributes)

        # Print text content
        text_content = einpresstechnik_locator.text_content()
        print("Text Content:", text_content)

        # Select all <h3> elements with the class 'headingClasses'
        locator = page.locator('h3.headingClasses')

        # Get the count of matching <h3> elements
        count = locator.count()
        print(f"Found {count} <h3> elements with class 'headingClasses'.")

        # Iterate over all matching elements and print their text content
        for i in range(count):
            text_content = locator.nth(i).text_content()
            print(f"Element {i + 1} text content: {text_content}")

        #=======================
        # Select all <a> elements with the class 'tile'
        locator = page.locator('a.tile')

        # Get the count of matching <a> elements
        count = locator.count()
        print(f"Found {count} <a> elements with class 'tile'.")

        # Iterate over all matching elements
        for i in range(count):
            # Get href attribute
            href = locator.nth(i).get_attribute('href')
            print(f"Element {i + 1} href: {href}")

            # Search for img.image--subcategory or img.image--category
            img_locator = locator.nth(i).locator('img.image--subcategory, img.image--category')
            category_type = None
            img_src = None
            if img_locator.count() > 0:
                img_src = img_locator.get_attribute('src')
                category_type = 'subcategory' if 'image--subcategory' in img_locator.get_attribute('class') else 'category'
                print(f"Element {i + 1} image URL: {img_src}")
            else:
                print(f"Element {i + 1} has no image with class 'image--subcategory' or 'image--category'.")

            # Search for h3.headingClasses and print the text content
            h3_locator = locator.nth(i).locator('h3.headingClasses')
            category_name = None
            if h3_locator.count() > 0:
                category_name = h3_locator.text_content().strip()
                print(f"Element {i + 1} H3 text content: {category_name}")
            else:
                print(f"Element {i + 1} has no H3 with class 'headingClasses'.")

            # Save image and prepare JSON entry
            if img_src and category_name:
                sanitized_name = sanitize_filename(category_name)
                # Download and save the image
                img_data = page.evaluate(f"""
                    async () => {{
                        const response = await fetch('{img_src}');
                        const buffer = await response.arrayBuffer();
                        return Array.from(new Uint8Array(buffer));
                    }}
                """)
                img_bytes = bytes(img_data)  # Convert list of integers to bytes
                img_path = f'./img/{sanitized_name}.jpeg'
                with open(img_path, 'wb') as img_file:
                    img_file.write(img_bytes)
                print(f"Image for element {i + 1} saved as {img_path}.")

                # Add entry to categories list
                categories.append({
                    "category_type": category_type,
                    "category_name": category_name,
                    "category_link": href,
                    "category_img": img_path
                })

        # Close browser
        browser.close()
        print("Script completed successfully!")

     # Save categories to JSON
    with open('categories.json', 'w', encoding='utf-8') as json_file:
        json.dump({"categories": categories}, json_file, ensure_ascii=False, indent=4)
    print("Categories saved to categories.json")

if __name__ == "__main__":
    main()
