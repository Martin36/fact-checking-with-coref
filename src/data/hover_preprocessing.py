import bz2, glob, json, re, pprint
from typing import List
from collections import defaultdict
from src.data.doc_db import DocDB
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=2)


def count_files(base_path):
  file_path = base_path + "/**/*.bz2"
  paths = glob.glob(file_path, recursive=True)
  return len(paths)


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
  new_db_path = "data/hover/wiki_with_lines.db"
  base_path = "data/hover/enwiki"

  db = DocDB(db_path=db_path)
  new_db = DocDB(db_path=new_db_path)
  
  # new_db.drop_documents_table()
  new_db.create_documents_table()
  
  nr_of_files = count_files(base_path)
  
  for idx, docs in enumerate(tqdm(data_loader(base_path), total=nr_of_files)):
    doc_titles = []
    doc_texts = []
    doc_lines = []
    
    # if idx < 9635:
    #   continue
    
    for doc in docs:
      doc_text = db.get_doc_text(doc["title"])      
      if not doc_text:
        # print(f"Doc {doc['title']} not found")
        continue
        
      lines_text = convert_lines_to_text(doc["text"])

      # new_db.insert_doc_with_lines(doc["title"], doc_text, lines_text)

      doc_titles.append(doc["title"])
      doc_texts.append(doc_text)
      doc_lines.append(lines_text)
            
    # doc_texts = db.get_multiple_docs_text(doc_titles)
    
    new_db.insert_docs_with_lines(doc_titles, doc_texts, doc_lines)
          
  print(f"Finished creating DB '{new_db_path}'")

