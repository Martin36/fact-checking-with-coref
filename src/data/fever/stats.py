
from collections import defaultdict
import os
from tqdm import tqdm
from utils_package.util_funcs import load_jsonl, store_json
from utils_package.logger import get_logger

from src.data.doc_db import DocDB
from src.utils.helpers import filter_empty, get_evidence_pages, get_fever_doc_lines

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
  
  
def get_list_of_long_docs(db_path, length_threshold=1000):
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
    
    if nr_of_sents > length_threshold:
      long_docs_list.append({
        "doc_id": doc_id,
        "nr_of_sents": nr_of_sents
      })
  
  return long_docs_list


def calculate_evidence_length_distribution(data_path):
  data = load_jsonl(data_path)
  evidence_length_counts = defaultdict(int)
  
  for d in data:
    if d["verifiable"] == "NOT VERIFIABLE":
      continue
    
    # Using the shortest evidence set      
    shortest_evidence_set_len = float("inf")
    for evidence_set in d["evidence"]:
      evidence_set_len = len(evidence_set)
      if evidence_set_len < shortest_evidence_set_len:
        shortest_evidence_set_len = evidence_set_len
    
    evidence_length_counts[shortest_evidence_set_len] += 1
    
  return evidence_length_counts


def calculate_claims_with_only_first_wiki_sent_as_evidence(data_path):
  data = load_jsonl(data_path)
  data = filter_verifiable(data)
  result = defaultdict(int)
  
  for d in data:    
    shortest_evidence_set = get_shortest_evidence_set(d["evidence"])
    
    if len(shortest_evidence_set) > 1:
      result["claims_with_more_than_one_evidence_sent"] += 1
      continue
    
    if is_first_evidence_first_sent_of_doc(shortest_evidence_set):
      result["claims_where_evidence_only_first_sent"] += 1
    
  result["verifiable_claims"] = len(data)
  result["claims_where_evidence_only_first_sent"] = result["claims_where_evidence_only_first_sent"] / result["verifiable_claims"]
  
  return result


def calculate_nr_of_sources_dist(data_path):
  data = load_jsonl(data_path)
  data = filter_verifiable(data)
  nr_of_sources_dist = []
  
  for d in data:        
    shortest_evidence_set = get_shortest_evidence_set(d["evidence"])
    evidence_pages = get_evidence_pages(shortest_evidence_set)
    nr_of_sources_dist.append(len(evidence_pages))
    
  return nr_of_sources_dist
  

def get_shortest_evidence_set(evidence_sets):
  shortest_evidence_set = None
  for evidence_set in evidence_sets:
    if not shortest_evidence_set:
      shortest_evidence_set = evidence_set 
    if len(evidence_set) < len(shortest_evidence_set):
      shortest_evidence_set = evidence_set
  return shortest_evidence_set

def calculate_avg_nr_of_sources(dist):
  return sum(dist) / len(dist)
  
def is_first_evidence_first_sent_of_doc(evidence_set):
  return evidence_set[0][3] == 0

def filter_verifiable(data):
  return [d for d in data if d["verifiable"] == "VERIFIABLE"]



if __name__ == "__main__":

  db_path = "data/fever/fever.db"
  train_data_path = "data/fever/train.jsonl"
  
  sent_idx_counts_file = "data/fever/stats/train_sent_idx_counts.json"
  doc_length_counts_file = "data/fever/stats/doc_length_counts.json"
  long_docs_file = "data/fever/stats/long_docs.json"
  evidence_len_dist_file = "data/fever/stats/evidence_length_distribution.json"
  claims_with_only_first_sent_evidence_file = "data/fever/stats/claims_with_only_first_sent_evidence.json"
  nr_of_sources_dist_file = "data/fever/stats/nr_of_sources_dist.json"

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

  if not os.path.isfile(evidence_len_dist_file):
    evidence_length_counts = calculate_evidence_length_distribution(train_data_path)
    store_json(evidence_length_counts, evidence_len_dist_file, sort_keys=True)
    logger.info(f"Stored evidence length counts in '{evidence_len_dist_file}'")
  else:
    logger.info(f"Evidence length counts already exists in '{evidence_len_dist_file}'")

  if not os.path.isfile(claims_with_only_first_sent_evidence_file):
    stats = calculate_claims_with_only_first_wiki_sent_as_evidence(train_data_path)
    store_json(stats, claims_with_only_first_sent_evidence_file)
    logger.info(f"Stored claims with only first wiki sent as evidence stats in '{claims_with_only_first_sent_evidence_file}'")
  else:
    logger.info(f"Claims with only first wiki sent as evidence stats already exists in '{claims_with_only_first_sent_evidence_file}'")

  if not os.path.isfile(nr_of_sources_dist_file):
    nr_of_sources_dist = calculate_nr_of_sources_dist(train_data_path)
    avg_nr_of_sources = calculate_avg_nr_of_sources(nr_of_sources_dist)
    store_json(nr_of_sources_dist, nr_of_sources_dist_file)
    logger.info(f"Stored number of sources distribution in '{nr_of_sources_dist_file}'")
    logger.info(f"Average number of sources: '{avg_nr_of_sources}'")
  else:
    logger.info(f"Number of sources distribution in '{nr_of_sources_dist_file}'")
    
  
