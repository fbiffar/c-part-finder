from playwright.sync_api import sync_playwright

def main():
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

        # Get the count of matching <h3> elements
        count = locator.count()
        print(f"Found {count} <h3> elements with class 'headingClasses'.")

        # Iterate over all matching elements and print their text content
        for i in range(count):
           href = locator.nth(i).get_attribute('href')
           print(f"Element {i + 1} href: {href}")


        # Close browser
        browser.close()
        print("Script completed successfully!")

if __name__ == "__main__":
    main()
