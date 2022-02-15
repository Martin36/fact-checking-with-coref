
import os
from tqdm import tqdm
from src.scraper.base import BaseScraper

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

from utils_package.util_funcs import store_jsonl, load_json, store_json
from utils_package.logger import get_logger

logger = get_logger()

class RiksdagenScraper(BaseScraper):

  def __init__(self) -> None:
    super().__init__()
    self.browser = webdriver.Firefox()
  
  
  def scrape_partiledardebatter(self):
    url = "https://www.riksdagen.se/sv/webb-tv/?doktyp=kam-pd"

    self.browser.get(url)
    
    hrefs_path = "data/riksdagen/hrefs-partiledardebatter.json"
    
    if os.path.isfile(hrefs_path):
      logger.info(f"Hrefs already stored in '{hrefs_path}', using those")
      hrefs = load_json(hrefs_path)
    else:
      hrefs = []
      hrefs += self.get_hrefs_from_page()
    
      next_button = self.browser.find_element(By.LINK_TEXT, "Nästa")
      try:
        while next_button.is_enabled():
          next_button.click()
          self.browser.refresh()
          hrefs += self.get_hrefs_from_page()
          next_button = self.browser.find_element(By.LINK_TEXT, "Nästa")
      except NoSuchElementException as e:
        print(e)
      finally:
        store_json(hrefs, hrefs_path)
        logger.info(f"Stored hrefs in '{hrefs_path}'")

    
    data = [self.scrape_statements(href) for href in tqdm(hrefs)]
    
    self.browser.close()
    
    return data
    
    
  def get_hrefs_from_page(self):
    main_container = WebDriverWait(self.browser, 10).until(
      EC.presence_of_element_located((By.CLASS_NAME, "search-list-webtv"))
    ) 
    link_elements = main_container.find_elements(By.LINK_TEXT, "Partiledardebatt")          
    hrefs = [elem.get_attribute("href") for elem in link_elements]    
    return hrefs    
  
  def scrape_statements(self, url):

    self.browser.get(url)
    self.browser.fullscreen_window()
    
    statements = []
    try:
      video_protocol_toggle = WebDriverWait(self.browser, 10).until(
        # EC.presence_of_element_located((By.ID, "video-protocol-toggle"))
        EC.element_to_be_clickable((By.ID, "video-protocol-toggle"))
      )
      video_protocol_toggle.click()
            
      transcript_container = self.browser.find_element(By.ID, "video-protocol")
      transcript_items = transcript_container.find_elements(By.CLASS_NAME, "video-transcript__item")
              
      for idx, item in enumerate(transcript_items):
        speaker_name = self.get_speaker_name(item)
        text = self.get_text(item)
        statement = {
          "index": idx, 
          "speaker": speaker_name,
          "text": text
        }
        statements.append(statement)
    except Exception as e:
      logger.error(e)

    date = self.get_date()
    
    result = {
      "url": url,
      "date": date,
      "statements": statements
    }
    
    return result    
    

  def get_speaker_name(self, item):
    name = item.find_element(By.TAG_NAME, "a").text.strip()
    return name
  
  def get_text(self, item):
    paragraph_texts = []
    paragraphs = item.find_elements(By.TAG_NAME, "p")
    if len(paragraphs) == 0:
      text = item.text.strip()
      return text
    
    for paragraph in paragraphs:
      text = paragraph.text.strip()
      paragraph_texts.append(text)
    result = "\n".join(paragraph_texts)
    return result

  def get_date(self):
    container = self.browser.find_element(By.CSS_SELECTOR, "article.page-content")
    first_row = container.find_elements(By.CLASS_NAME, "row")[0]
    date = first_row.find_element(By.TAG_NAME, "p").text.strip()
    return date
    
    
if __name__ == "__main__":
  
  scraper = RiksdagenScraper()
  data = scraper.scrape_partiledardebatter()

  out_file = "data/riksdagen/partiledardebatter.jsonl"
  store_jsonl(data, out_file)
  print(f"Stored data in '{out_file}'")  
  
  # url = "https://www.riksdagen.se/sv/webb-tv/video/partiledardebatt/partiledardebatt_GZC120120613pd"
  # data = scraper.scrape_statements(url)
  # print(data)
