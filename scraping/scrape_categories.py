import os
import json
from playwright.sync_api import sync_playwright

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

            child_elements = locator.nth(i).locator('> *')
            child_count = child_elements.count()
            print(f"Element {i + 1} has {child_count} direct child elements.")
            for j in range(child_count):
                tag_name = child_elements.nth(j).evaluate("el => el.tagName")
                class_name = child_elements.nth(j).get_attribute('class')
                print(f"Child {j + 1}: Tag Name = {tag_name}, Class = {class_name}")

            #  Search for img.image--subcategory and print the media URL
            img_locator = locator.nth(i).locator('img.image--subcategory, img.image--category')
            if img_locator.count() > 0:
                img_src = img_locator.get_attribute('src')
                print(f"Element {i + 1} image URL: {img_src}")
            else:
                print(f"Element {i + 1} has no image with class 'image--subcategory'.")

            # Search for h3.headingClasses and print the text content
            h3_locator = locator.nth(i).locator('h3.headingClasses')
            if h3_locator.count() > 0:
                h3_text = h3_locator.text_content()
                print(f"Element {i + 1} H3 text content: {h3_text}")
            else:
                print(f"Element {i + 1} has no H3 with class 'headingClasses'.")


        # Close browser
        browser.close()
        print("Script completed successfully!")

if __name__ == "__main__":
    main()
