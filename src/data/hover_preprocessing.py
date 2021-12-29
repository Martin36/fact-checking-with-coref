import bz2, glob, json, re
from typing import List
from src.data.doc_db import DocDB


def data_loader(base_path):
  file_path = base_path + "/**/*.bz2"
  paths = glob.glob(file_path, recursive=True)
  for path in paths:
    with bz2.open(path, "rb") as f:
      content = f.read().decode("utf-8")
      lines = content.split("\n")
      data = [json.loads(line) for line in lines if len(line) > 0]
      yield data
    

def remove_hyperlinks(sent: str):
  pattern = r"<a href=\"[^\"]*\">|<\/a>"
  sent = re.sub(pattern, "", sent)
  return sent
  

def escape_single_quotes(sent: str):
  sent = sent.replace("\'", "\'\'")
  return sent


def process_sentence(sent: str):
  sent = sent.strip()
  sent = remove_hyperlinks(sent)
  sent = escape_single_quotes(sent)
  return sent
    

def convert_lines_to_text(paragraphs: List[List[str]]):
  """Converts the Wiki dump format to a concatenated string

  Args:
      paragraphs (List[List[str]]): Each list in the list represents a paragraph from the document
  """
  
  paragraph_texts = []
  for i, paragraph in enumerate(paragraphs):
    if i == 0:
      # The first paragraph seems to be the title
      # TODO: Does skipping this fuck up the order?
      continue
      
    paragraph_sents = []
    for sent in paragraph:    
      sent = process_sentence(sent)
      paragraph_sents.append(sent)
    paragraph_text = "\n".join(paragraph_sents)
    paragraph_texts.append(paragraph_text)
  
  paragraph_texts = [t for t in paragraph_texts if len(t) > 0]
  doc_text = "\n".join(paragraph_texts)
  return doc_text
      

if __name__ == "__main__":
  db_path = "data/hover/wiki_wo_links.db"
  db = DocDB(db_path=db_path)
  
  bz2_file = "data/hover/enwiki/AA/wiki_00.bz2"

  base_path = "data/hover/enwiki"
  
  for docs in data_loader(base_path):
    for doc in docs:
      lines_text = convert_lines_to_text(doc["text"])
      
      db.store_doc_lines(doc["title"], lines_text)
      
      print(doc)
  
  

