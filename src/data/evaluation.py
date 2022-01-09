from typing import List
from collections import defaultdict
from utils_package.util_funcs import calc_f1, load_jsonl
from src.utils.helpers import get_evidence_pages

from src.utils.types import DocRetrievalResult

stats = defaultdict(int)

def evaluate_doc_retrieval(data: List[DocRetrievalResult], include_nei=False):
  
  accuracy, precision, recall, f1 = 0, 0, 0, 0
  
  if not include_nei:
    data = [d for d in data if d["verifiable"] == "VERIFIABLE"]
    
  for d in data:
    
    if len(d["predicted_pages"]) == 0:
      stats["no_predicted_pages"] += 1
      
    precision_d, recall_d, accuracy_d = 0, 0, 0
    
    evidence_sents = d["evidence"][0]   # TODO: How to handle multiple evidence?
    evidence_pages = get_evidence_pages(evidence_sents)
    
    for page in evidence_pages:
      if page in d["predicted_pages"]:
        accuracy_d += 1
        recall_d += 1
        precision_d += 1
        
    accuracy_d /= len(evidence_pages)
    recall_d /= len(evidence_pages)
    if len(d["predicted_pages"]) > 0:    
      precision_d /= len(d["predicted_pages"])
    else:
      precision_d = 1.0
    
    accuracy += accuracy_d
    recall += recall_d
    precision += precision_d

  accuracy /= len(data)        
  recall /= len(data)        
  precision /= len(data)        
  f1 = calc_f1(precision, recall)
  
  result = {
    "accuracy": accuracy,
    "recall": recall,
    "precision": precision,
    "f1": f1
  }
  
  return result
  
if __name__ == "__main__":
  
  data_file = "data/fever/doc_retrieval/train.wiki7.jsonl"
  data = load_jsonl(data_file)
  
  metrics = evaluate_doc_retrieval(data)
  
  print(metrics)
  
  print(stats)