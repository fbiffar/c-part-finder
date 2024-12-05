import os
import logging
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BossardScraper:
    def __init__(self):
        self.base_url = "https://www.bossard.com/eshop/ch-de/"
        self.visited_urls = set()
        self.playwright = None
        self.browser = None
        self.context = None
        self.img_dir = './img/'
        self.links_file = './links/links.txt'

    def initialize(self):
        """Initialize Playwright browser"""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        logger.info("Playwright initialized successfully")

    def get_page_content(self, url):
        """Load page and wait for dynamic content"""
        try:
            page = self.context.new_page()
            page.goto(url, wait_until='networkidle')
            return page
        except Exception as e:
            logger.error(f"Error loading page {url}: {e}")
            return None

    def scrape_media_images(self, page):
        """Scrape all media images from the page and save them locally"""
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        image_elements = page.query_selector_all('img')  # Adjust selector if needed

        for img in image_elements:
            img_url = img.get_attribute('src')
            if img_url and 'medias' in img_url:  # Filter for media images
                self.download_image(img_url)

    def download_image(self, img_url):
        """Download an image from a URL and save it to the img directory"""
        try:
            img_url = urljoin(self.base_url, img_url)
            img_name = os.path.basename(urlparse(img_url).path)
            img_path = os.path.join(self.img_dir, img_name)

            # Use Playwright to download the image
            page = self.context.new_page()
            response = page.goto(img_url)
            if response.ok:
                with open(img_path, 'wb') as f:
                    f.write(response.body())
                logger.info(f"Downloaded image: {img_name}")
            else:
                logger.error(f"Failed to download image {img_url}: HTTP {response.status}")

            page.close()

        except Exception as e:
            logger.error(f"Failed to download image {img_url}: {e}")

    def scrape_links(self, page):
        """Scrape all links from the page and save them to a file"""
        if not os.path.exists(os.path.dirname(self.links_file)):
            os.makedirs(os.path.dirname(self.links_file))

        link_elements = page.query_selector_all('a')
        with open(self.links_file, 'w', encoding='utf-8') as f:
            for link in link_elements:
                href = link.get_attribute('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    f.write(full_url + '\n')
                    logger.info(f"Found link: {full_url}")

    def scrape(self):
        """Main scraping method"""
        try:
            self.initialize()
            logger.info("Starting scraping process...")
            
            page = self.get_page_content(self.base_url)
            if not page:
                return None
            
            self.scrape_media_images(page)
            self.scrape_links(page)
            page.close()

            logger.info("Scraping completed successfully!")

        except Exception as e:
            logger.error(f"Scraping failed: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup Playwright resources"""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

def main():
    scraper = BossardScraper()
    scraper.scrape()

if __name__ == "__main__":
    main()