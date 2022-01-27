

from src.scraper.base import BaseScraper
import requests
from bs4 import BeautifulSoup

class PolitiFactScraper(BaseScraper):
  
  def __init__(self, url) -> None:
    super().__init__()
    page = requests.get(url)
    self.soup = BeautifulSoup(page.content, "html.parser")


  def scrape_categories(self):
    base_url = "https://www.politifact.com"
    url = "https://www.politifact.com/issues/"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    category_sections = soup.find_all("section", 
      {"id": lambda l: l and l.startswith("issues_")})
    results = []
    
    for section in category_sections:
      categories = section.find_all("a")
    
      for category in categories:
        href = category["href"]
        name = category.text.strip()
        results.append({
          "name": name,
          "link": base_url + href
        })
    
    return results
    
    
  
  def scrape_sources(self):
    sources_container = self.soup.find(id="sources")
    sources = sources_container.find_all("p")
    sources_list = []
    for source in sources:
      link = source.find("a")
      link_href = None
      if link: 
        link_href = link["href"]
      source_obj = {
        "text": source.text.strip(),
        "link": link_href
      }
      sources_list.append(source_obj)
    return sources_list
    
  
    
if __name__ == "__main__":
  
  url = "https://www.politifact.com/factchecks/2022/jan/13/joe-biden/evidence-scant-joe-biden-was-arrested-protesting-c/"
  politi_fact_scraper = PolitiFactScraper(url)
  # source_list = politi_fact_scraper.scrape_sources()
  
  categories = politi_fact_scraper.scrape_categories()