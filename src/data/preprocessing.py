import numpy as np

from typing import List
from utils_package.util_funcs import load_jsonl, store_jsonl
from utils_package.logger import get_logger

from src.data.doc_db import DocDB
from src.data.fever_dataset import FeverDataSample
from src.utils.helpers import get_evidence_pages, get_fever_doc_lines
from src.utils.types import DocumentLevelFever

logger = get_logger()

    
    
def filter_empty_sents(d: DocumentLevelFever):
  non_empty_sents_indices = [i for i, sent in enumerate(d["sentences"]) 
                             if len(sent) > 0]
  d["sentences"] = [d["sentences"][i] for i in non_empty_sents_indices]
  d["label_list"] = [d["label_list"][i] for i in non_empty_sents_indices]
  d["sentence_IDS"] = [d["sentence_IDS"][i] for i in non_empty_sents_indices]
  return d
  

def create_document_level_fever_data(db: DocDB, data: FeverDataSample) -> List[DocumentLevelFever]:
  result = []
  for d in data:
    # Skip the NEI samples since these don't have any evidence to retrieve
    if d["verifiable"] == "NOT VERIFIABLE":
      continue
    
    evidence_sents = d["evidence"][0] # TODO: How to handle the other evidence sets?
    evidence_pages = get_evidence_pages(evidence_sents)
    
    for page in evidence_pages:
      res_obj = DocumentLevelFever()
      res_obj["id"] = d["id"]
      res_obj["claim"] = d["claim"]
      res_obj["page"] = page

      doc_text = db.get_doc_lines(page)
      doc_lines = get_fever_doc_lines(doc_text)
      res_obj["sentences"] = doc_lines
      
      gold_sent_indexes = [evidence[3] for evidence in evidence_sents 
                           if evidence[2] == page]
      res_obj["label_list"] = [int(i in gold_sent_indexes) 
                               for i in range(len(doc_lines))]
      
      res_obj["sentence_IDS"] = np.arange(len(doc_lines)).tolist()
            
      res_obj = filter_empty_sents(res_obj)
      
      result.append(res_obj)
      
  return result
  
  
if __name__ == "__main__":

  db_path = "data/fever/fever.db"
  db = DocDB(db_path=db_path)

  dev_path = "data/fever/dev_with_evidence.jsonl"
  data = load_jsonl(dev_path)
  
  document_level_fever_data = create_document_level_fever_data(db, data)
  
  out_file = "data/fever/dev_document_level_fever.jsonl"
  store_jsonl(document_level_fever_data, out_file)
  logger.info(f"Stored document level fever data in '{out_file}'")