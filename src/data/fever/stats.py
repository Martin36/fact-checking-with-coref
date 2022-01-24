
from collections import defaultdict
import os
from tqdm import tqdm
from utils_package.util_funcs import load_jsonl, store_json
from utils_package.logger import get_logger

from src.data.doc_db import DocDB
from src.utils.helpers import filter_empty, get_fever_doc_lines

logger = get_logger()


def calculate_sentence_index_distribution(data_path):
  data = load_jsonl(data_path)
  sent_idx_counts = defaultdict(int)
  
  for d in data:
    if d["verifiable"] == "NOT VERIFIABLE":
      continue
    
    # TODO: Count all evidence sets or just the first one (or a randomly selected)?
    for evidence_set in d["evidence"]:
      for evidence in evidence_set:      
        sent_idx = evidence[3]
        sent_idx_counts[sent_idx] += 1
  
  return sent_idx_counts
        

def calculate_wiki_doc_sentence_lengths(db_path):
  db = DocDB(db_path)
  doc_ids = db.get_doc_ids()
  doc_lengths = defaultdict(int)
  
  for doc_id in tqdm(doc_ids):
    doc_text = db.get_doc_lines(doc_id)
    if not doc_text or len(doc_text) == 0:
      continue
    doc_lines = get_fever_doc_lines(doc_text)
    doc_lines = filter_empty(doc_lines)
    nr_of_sents = len(doc_lines)
    doc_lengths[nr_of_sents] += 1
      
  return doc_lengths
  
  
def get_list_of_long_docs(db_path, length_threshold):
  db = DocDB(db_path)
  doc_ids = db.get_doc_ids()
  long_docs_list = []
  
  for doc_id in tqdm(doc_ids):
    doc_text = db.get_doc_lines(doc_id)
    if not doc_text or len(doc_text) == 0:
      continue
    doc_lines = get_fever_doc_lines(doc_text)
    doc_lines = filter_empty(doc_lines)
    nr_of_sents = len(doc_lines)
    
    if nr_of_sents > 1000:
      long_docs_list.append({
        "doc_id": doc_id,
        "nr_of_sents": nr_of_sents
      })
  
  return long_docs_list



if __name__ == "__main__":

  db_path = "data/fever/fever.db"
  train_data_path = "data/fever/train.jsonl"
  
  sent_idx_counts_file = "data/fever/stats/train_sent_idx_counts.json"
  doc_length_counts_file = "data/fever/stats/doc_length_counts.json"
  long_docs_file = "data/fever/stats/long_docs.json"

  if not os.path.isfile(sent_idx_counts_file):
    sent_idx_counts = calculate_sentence_index_distribution(train_data_path)
    store_json(sent_idx_counts, sent_idx_counts_file, sort_keys=True)
    logger.info(f"Stored sentence index counts data in '{sent_idx_counts_file}'")
  else:
    logger.info(f"Sentence index counts already exists in '{sent_idx_counts_file}'")
    
  if not os.path.isfile(doc_length_counts_file):
    doc_length_counts = calculate_wiki_doc_sentence_lengths(db_path)
    store_json(doc_length_counts, doc_length_counts_file, sort_keys=True)
    logger.info(f"Stored document length counts data in '{doc_length_counts_file}'")
  else:
    logger.info(f"Document length counts already exists in '{doc_length_counts_file}'")

  if not os.path.isfile(long_docs_file):
    length_threshold = 1000   # Determines what is considered a long document
    long_docs = get_list_of_long_docs(db_path, length_threshold)
    store_json(long_docs, long_docs_file, sort_keys=True)
    logger.info(f"Stored long docs in '{long_docs_file}'")
  else:
    logger.info(f"Long docs already exists in '{long_docs_file}'")
