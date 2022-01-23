import argparse
from collections import defaultdict
import random
import numpy as np

from tqdm import tqdm
from utils_package.util_funcs import load_jsonl, store_json, store_jsonl
from utils_package.logger import get_logger

from src.data.doc_db import DocDB
from src.utils.helpers import get_evidence_pages, get_fever_doc_lines, get_non_empty_indices, get_random_from_list

logger = get_logger()

pages_not_found = set()
stats = defaultdict(int)
        
  
class DocumentLevelFeverPreprocessor():
  
  def __init__(self, db_path, retrieved_docs_file):
    self.db = DocDB(db_path=db_path)
    self.retrieved_docs_data = load_jsonl(
      retrieved_docs_file
    )
    self.is_data_unlabelled = self.check_if_unlabelled_data()
    self.max_nr_of_docs = 5


  def check_if_unlabelled_data(self):
    # Assuming that if the first element in the dataset determines that whole dataset
    if "verifiable" in self.retrieved_docs_data[0]:
      return False
    return True
        
  
  def create_data_from_retrieved_docs(self):
    result = []    
    if self.is_data_unlabelled:
      result = self.create_unlabelled_data_from_retrieved_docs()
    else:
      result = self.create_labelled_data_from_retrieved_docs()
    return result
  
  
  def create_unlabelled_data_from_retrieved_docs(self):
    result = []
    for d in tqdm(self.retrieved_docs_data):
      evidence_sents, evidence_pages = None, None
      evidence_pages = d["predicted_pages"]
      random.shuffle(evidence_pages)
      evidence_pages = evidence_pages[:self.max_nr_of_docs]
      result += self.create_data_obj_from_evidence(evidence_pages, evidence_sents, d)
    return result
    
  
  def create_labelled_data_from_retrieved_docs(self):
    result = []
    for d in tqdm(self.retrieved_docs_data):
      evidence_sents, evidence_pages = None, None
      if d["verifiable"] == "NOT VERIFIABLE":
        evidence_pages = d["predicted_pages"]
        random.shuffle(evidence_pages)
        evidence_pages = evidence_pages[:self.max_nr_of_docs]
      else:
        evidence_pages = self.get_pages_from_gold_evidence(d)
      result += self.create_data_obj_from_evidence(evidence_pages, evidence_sents, d)
    return result
    

  def get_pages_from_gold_evidence(self, d):
    evidence_sents = d["evidence"][0] # TODO: How to handle the other evidence sets?
    evidence_pages = get_evidence_pages(evidence_sents)
    evidence_pages = self.add_retrieved_pages(
      d["predicted_pages"], evidence_pages
    )
    return evidence_pages


  def add_retrieved_pages(self, predicted_pages, evidence_pages):
    # Remove the gold pages before, so we don't bias towards them
    predicted_pages = [page for page in predicted_pages 
                       if page not in evidence_pages]
    predicted_pages = get_random_from_list(predicted_pages)
    result = evidence_pages + predicted_pages
    # make sure that the gold evidence pages are kept by trimming
    # the list from the end
    result = result[:self.max_nr_of_docs]
    random.shuffle(result)
    return result


  def create_data_objs_from_evidence(self, evidence_pages, evidence_sents, d):
    result = []
    if len(evidence_pages) == 0:
      stats["claims_without_evidence_pages"] += 1
      evidence_pages = get_random_from_list(
        d["wiki_results"], max_n=self.max_nr_of_docs
      )
  
    for page in evidence_pages:
      res_obj = self.create_document_level_fever_obj(
        page, evidence_sents, d["id"], d["claim"]
      )
      if not res_obj:
        continue
      result.append(res_obj)
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

    res_obj = dict()
    res_obj["id"] = id
    res_obj["claim"] = claim
    res_obj["page"] = page
    res_obj["sentences"] = doc_lines
    res_obj["label_list"] = [int(i in gold_sent_indexes) 
                            for i in range(len(doc_lines))]
    res_obj["sentence_IDS"] = np.arange(len(doc_lines)).tolist()
          
    res_obj = self.filter_empty_sents(res_obj)
    return res_obj

  
  def filter_empty_sents(self, d):
    non_empty_sents_indices = get_non_empty_indices(d["sentences"])
    d["sentences"] = [d["sentences"][i] for i in non_empty_sents_indices]
    d["label_list"] = [d["label_list"][i] for i in non_empty_sents_indices]
    d["sentence_IDS"] = [d["sentence_IDS"][i] for i in non_empty_sents_indices]
    return d


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--retrieved_docs_file', type=str, help="Path to the dataset containing the retrieved documents")
  parser.add_argument('--pages_not_found_file', type=str, default=None, help="(Optional) Path to the file to store pages that are not found")
  parser.add_argument('--out_file', type=str, help="Path to the file to store the output data, which in this case will be the input data to the Document Level FEVER sentence retrieval")
  args = parser.parse_args()

  db_path = "data/fever/fever.db"
  db = DocDB(db_path=db_path)
    
  preprocessor = DocumentLevelFeverPreprocessor(
    db_path, args.retrieved_docs_file
  )
  
  output_data = preprocessor.create_data_from_retrieved_docs()
  logger.info(f"Nr of pages not found: {len(pages_not_found)}")
  if args.pages_not_found_file:
    store_json(list(pages_not_found), args.pages_not_found_file, indent=2)
  
  store_jsonl(output_data, args.out_file)
  logger.info(f"Stored document level fever train data in '{args.out_file}'")
  
  print("stats: ", stats)
