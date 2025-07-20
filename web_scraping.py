import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import random
import time

class PoliteSpider(scrapy.Spider):
    name = "polite"
    
    # Custom settings for this spider
    custom_settings = {
        'DOWNLOAD_DELAY': 2,  # Default 2 second delay
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1.0,
        'AUTOTHROTTLE_MAX_DELAY': 10.0,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'ROBOTSTXT_OBEY': True,
    }
    
    def __init__(self, start_url=None, max_pages=10, delay=None, *args, **kwargs):
        super(PoliteSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url] if start_url else ['https://example.com']
        self.max_pages = max_pages
        self.visited_pages = 0
        if delay:
            self.custom_settings['DOWNLOAD_DELAY'] = float(delay)

    def parse(self, response):
        self.visited_pages += 1
        
        # Extract data
        title = response.css('title::text').get()
        links = response.css('a::attr(href)').getall()
        
        yield {
            'url': response.url,
            'title': title,
            'links': links,
            'visited_pages': self.visited_pages
        }
        
        # Follow links if we haven't reached max pages
        if self.visited_pages < self.max_pages:
            for link in links:
                if link.startswith(('http://', 'https://')):
                    # Add random delay between 0.5x and 1.5x of DOWNLOAD_DELAY
                    time.sleep(random.uniform(
                        0.5 * self.custom_settings['DOWNLOAD_DELAY'],
                        1.5 * self.custom_settings['DOWNLOAD_DELAY']
                    ))
                    yield response.follow(link, callback=self.parse)

class RandomDelayMiddleware:
    """Custom middleware for additional random delays"""
    def process_request(self, request, spider):
        delay = random.uniform(0.5, 1.5)  # Random delay between 0.5-1.5 seconds
        time.sleep(delay)
        return None