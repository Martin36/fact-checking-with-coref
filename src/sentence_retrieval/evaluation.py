from collections import defaultdict
from ast import literal_eval
from tqdm import tqdm
import os
from utils_package.util_funcs import load_jsonl, calc_f1, store_json
import pandas as pd

from src.utils.enums import FeverLabels


stats = defaultdict(int)

def get_predicted_sentences(df):
  pred_sents = []
  for _, row in df.iterrows():
    if row["y"] == 1:
      sent_tuple = literal_eval(row["page_sentence"])
      pred_sents.append(sent_tuple)
  return pred_sents


def evaluate_document_level_fever(labelled_data_file: str, 
                                  output_data_file: str, 
                                  include_nei=False):
  """Evaluates the predictions made by document level fever

  Args:
      labelled_data_file (str): 
        The original FEVER dataset file e.g. dev.jsonl
      output_data_file (str): 
        The output file of the document level fever model
      include_nei (bool, optional): 
        Set this to true if the NEI samples should be included in 
        the evaluation e.g. if the evaluation should penalize the 
        model for predicting many sentences for NEI claims. 
        Defaults to False.

  Returns:
      [type]: [description]
  """
    
  labelled_data = load_jsonl(labelled_data_file)
  output_data = pd.read_csv(output_data_file)
  
  accuracy, precision, recall, f1 = 0, 0, 0, 0

  if not include_nei:
    labelled_data = [d for d in labelled_data 
                     if d["label"] != FeverLabels.NOT_ENOUGH_INFO.value]
    
  for d in tqdm(labelled_data):
        
    claim_rows = output_data.loc[output_data["claim_id"] == d["id"]]
    predicted_sents = get_predicted_sentences(claim_rows)
    
    if len(predicted_sents) == 0:
      stats["no_predicted_sentences"] += 1
      
    precision_d, recall_d, accuracy_d = 0, 0, 0
    
    evidence_sents = d["evidence"][0]   # TODO: How to handle multiple evidence?
    
    for evidence_sent in evidence_sents:
      for pred_sent in predicted_sents:         
        if evidence_sent[2] == pred_sent[0] and \
           evidence_sent[3] == pred_sent[1]:
          accuracy_d += 1
          recall_d += 1
          precision_d += 1
    
    if include_nei:
      # For NEI samples the accuracy and recall will always be 1
      # since there is no evidence to retrieve
      accuracy_d = 1.0
      recall_d = 1.0
    else:  
      accuracy_d /= len(evidence_sents)
      recall_d /= len(evidence_sents)
      
    if len(predicted_sents) > 0:
      precision_d /= len(predicted_sents)
    else:
      precision_d = 1.0
    
    accuracy += accuracy_d
    recall += recall_d
    precision += precision_d

  accuracy /= len(labelled_data)
  recall /= len(labelled_data)
  precision /= len(labelled_data)
  f1 = calc_f1(precision, recall)
  
  result = {
    "accuracy": accuracy,
    "recall": recall,
    "precision": precision,
    "f1": f1
  }
  
  return result

  
if __name__ == "__main__":
  labelled_data_file = "data/fever/dev.jsonl"
  output_data_file = "data/fever/predictions_sentence_retrieval.csv"
  results_file = "data/fever/sentence_retrieval/document_level_fever.json"
  
  metrics = evaluate_document_level_fever(labelled_data_file, output_data_file)
  
  results_folder = os.path.dirname(results_file)
  if not os.path.exists(results_folder):
    os.makedirs(results_folder)
  store_json(metrics, results_file, indent=2)
  
  print(f"Stored results in '{results_file}'")
  
  
  