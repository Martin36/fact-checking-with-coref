from collections import defaultdict
import random
import numpy as np

from typing import List
from tqdm import tqdm
from utils_package.util_funcs import load_jsonl, store_json, store_jsonl
from utils_package.logger import get_logger

from src.data.doc_db import DocDB
from src.utils.helpers import get_evidence_pages, get_fever_doc_lines, get_random_from_list
from src.utils.types import DocRetrievalResult, DocumentLevelFever

logger = get_logger()

pages_not_found = set()
stats = defaultdict(int)
    

def filter_empty_sents(d: DocumentLevelFever):
  non_empty_sents_indices = [i for i, sent in enumerate(d["sentences"]) 
                             if len(sent) > 0]
  d["sentences"] = [d["sentences"][i] for i in non_empty_sents_indices]
  d["label_list"] = [d["label_list"][i] for i in non_empty_sents_indices]
  d["sentence_IDS"] = [d["sentence_IDS"][i] for i in non_empty_sents_indices]
  return d
    
  
class DocumentLevelFeverPreprocessor():
  
  def __init__(self, db_path, retrieved_docs_file):
    self.db = DocDB(db_path=db_path)
    self.retrieved_docs_data: List[DocRetrievalResult] = load_jsonl(
      retrieved_docs_file
    )
    

  def add_retrieved_pages(self, predicted_pages, evidence_pages, max_length=5):
    # Remove the gold pages before, so we don't bias towards them
    predicted_pages = [page for page in predicted_pages 
                       if page not in evidence_pages]
    predicted_pages = get_random_from_list(predicted_pages)
    result = evidence_pages + predicted_pages
    if len(result) > max_length:
      # make sure that the gold evidence pages are kept by trimming
      # the list from the end
      result = result[:max_length]
    random.shuffle(result)
    return result
  

  def create_document_level_fever_obj(self, page, evidence_sents, id, claim):
    doc_text = self.db.get_doc_lines(page)
    if not doc_text:
      pages_not_found.add(page)
      return None
    
    doc_lines = get_fever_doc_lines(doc_text)
    
    if not evidence_sents:
      gold_sent_indexes = []
    else:
      gold_sent_indexes = [evidence[3] for evidence in evidence_sents 
                          if evidence[2] == page]

    res_obj = DocumentLevelFever()
    res_obj["id"] = id
    res_obj["claim"] = claim
    res_obj["page"] = page
    res_obj["sentences"] = doc_lines
    res_obj["label_list"] = [int(i in gold_sent_indexes) 
                            for i in range(len(doc_lines))]
    res_obj["sentence_IDS"] = np.arange(len(doc_lines)).tolist()
          
    res_obj = filter_empty_sents(res_obj)
    return res_obj
    
  
  def create_data_from_retrieved_docs(self):
    result = []
    max_nr_of_docs = 5
    
    for d in tqdm(self.retrieved_docs_data):
      
      evidence_sents, evidence_pages = None, None
      if d["verifiable"] == "NOT VERIFIABLE":
        evidence_pages = d["predicted_pages"]
        random.shuffle(evidence_pages)
        evidence_pages = evidence_pages[:max_nr_of_docs]
      else:
        evidence_sents = d["evidence"][0] # TODO: How to handle the other evidence sets?
        evidence_pages = get_evidence_pages(evidence_sents)
        evidence_pages = self.add_retrieved_pages(
          d["predicted_pages"], evidence_pages
        )
      
      if len(evidence_pages) == 0:
        stats["claims_without_evidence_pages"] += 1
        evidence_pages = get_random_from_list(
          d["wiki_results"], max_n=max_nr_of_docs
        )
      
      for page in evidence_pages:
        res_obj = self.create_document_level_fever_obj(
          page, evidence_sents, d["id"], d["claim"]
        )
        if not res_obj:
          continue
        result.append(res_obj)
        
    return result
  
  def create_data_from_gold_evidence(self) -> List[DocumentLevelFever]:
    result = []
    for d in data:
      # Skip the NEI samples since these don't have any evidence to retrieve
      if d["verifiable"] == "NOT VERIFIABLE":
        continue
      
      evidence_sents = d["evidence"][0] # TODO: How to handle the other evidence sets?
      evidence_pages = get_evidence_pages(evidence_sents)
      
      for page in evidence_pages:
        res_obj = self.create_document_level_fever_obj(
          page, evidence_sents, d["id"], d["claim"]
        )
        if not res_obj:
          continue
        result.append(res_obj)
        
    return result

  
  
if __name__ == "__main__":

  db_path = "data/fever/fever.db"
  db = DocDB(db_path=db_path)

  dev_path = "data/fever/dev_with_evidence.jsonl"
  data = load_jsonl(dev_path)
  
  retrieved_docs_file = "data/fever/doc_retrieval/train.wiki7.jsonl"
  pages_not_found_file = "data/fever/doc_retrieval/pages_not_found.json"
  
  preprocessor = DocumentLevelFeverPreprocessor(
    db_path, retrieved_docs_file
  )
  
  output_data = preprocessor.create_data_from_retrieved_docs()
  logger.info(f"Nr of pages not found: {len(pages_not_found)}")
  store_json(list(pages_not_found), pages_not_found_file, indent=2)
  
  out_file = "data/fever/train_document_level_fever.jsonl"
  store_jsonl(output_data, out_file)
  logger.info(f"Stored document level fever train data in '{out_file}'")
  
  print("stats: ", stats)
  # document_level_fever_data = create_document_level_fever_data(db, data)
  
  # out_file = "data/fever/dev_document_level_fever.jsonl"
  # store_jsonl(document_level_fever_data, out_file)
  # logger.info(f"Stored document level fever data in '{out_file}'")