from tqdm import tqdm
import os
from utils_package.util_funcs import load_jsonl, store_jsonl
import pandas as pd

from src.data.doc_db import DocDB
from src.utils.dlf_helpers import get_predicted_sentences
from src.utils.helpers import get_evidence_texts


# dlf = document level fever

def extract_evidence_from_predicted_sents(predicted_sents):
  return [[[None, None, doc_id, sent_id] 
            for doc_id, sent_id in predicted_sents]]


def merge_with_fever_data(labelled_data_file: str, 
                          output_data_file: str, db_path: str):
  
  db = DocDB(db_path)
  labelled_data = load_jsonl(labelled_data_file)
  output_data = pd.read_csv(output_data_file)
      
  for d in tqdm(labelled_data):
        
    claim_rows = output_data.loc[output_data["claim_id"] == d["id"]]
    predicted_sents = get_predicted_sentences(claim_rows)
    
    d["evidence"] = extract_evidence_from_predicted_sents(predicted_sents)
    d["evidence_texts"] = get_evidence_texts(db, d)
          
  return labelled_data

  
if __name__ == "__main__":
  db_path = "data/fever/fever.db"
  labelled_data_file = "data/fever/dev.jsonl"
  output_data_file = "data/fever/predictions_sentence_retrieval.csv"
  results_file = "data/fever/dev_with_dlf.jsonl"
  
  merged_data = merge_with_fever_data(labelled_data_file, 
                                      output_data_file, 
                                      db_path)
  
  results_folder = os.path.dirname(results_file)
  if not os.path.exists(results_folder):
    os.makedirs(results_folder)
  store_jsonl(merged_data, results_file)
  
  print(f"Stored results in '{results_file}'")
  
  
  