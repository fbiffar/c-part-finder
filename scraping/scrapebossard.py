import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BossardScraper:
    def __init__(self):
        self.base_url = "https://www.bossard.com/eshop/ch-de/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'de,en-US;q=0.7,en;q=0.3',
        })
        self.visited_urls = set()

    def initialize_session(self):
        """Initialize session and get necessary cookies"""
        try:
            response = self.session.get(self.base_url)
            response.raise_for_status()
            logger.info("Session initialized successfully")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to initialize session: {e}")
            return False

    def get_page_content(self, url):
        """Fetch page content with retry mechanism"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                time.sleep(1)  # Rate limiting
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for URL {url}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        return None

    def parse_category_page(self, url):
        """Parse a category page and extract subcategories"""
        if url in self.visited_urls:
            return {}
        
        self.visited_urls.add(url)
        logger.info(f"Parsing category page: {url}")
        
        content = self.get_page_content(url)
        print(content)
        if not content:
            return {}

        soup = BeautifulSoup(content, 'html.parser')
        categories = {}

        # Look for category elements (adjust selectors based on actual HTML structure)
        category_elements = soup.select('.category-element')  # Replace with actual selector
        
        for element in category_elements:
            category_name = element.get_text(strip=True)
            category_url = urljoin(self.base_url, element.get('href', ''))
            
            if category_url and category_url not in self.visited_urls:
                subcategories = self.parse_category_page(category_url)
                categories[category_name] = subcategories

        return categories

    def scrape(self):
        """Main scraping method"""
        if not self.initialize_session():
            logger.error("Failed to initialize session. Exiting...")
            return None

        logger.info("Starting scraping process...")
        categories = self.parse_category_page(self.base_url)

        # Save results to JSON file
        try:
            with open('bossard_categories.json', 'w', encoding='utf-8') as f:
                json.dump(categories, f, ensure_ascii=False, indent=4)
            logger.info("Categories saved to bossard_categories.json")
        except IOError as e:
            logger.error(f"Failed to save categories to file: {e}")

        return categories

def main():
    scraper = BossardScraper()
    categories = scraper.scrape()
    if categories:
        print("Scraping completed successfully!")
        print(f"Total categories visited: {len(scraper.visited_urls)}")

if __name__ == "__main__":
    main()