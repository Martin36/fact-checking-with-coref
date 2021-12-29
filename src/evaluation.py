import argparse
from enum import Enum
from tqdm import tqdm

from utils_package import util_funcs
from utils_package import get_logger

logger = get_logger()

class EvalTypes(Enum):
  DocRetrieval = "document_retrieval"
  

def eval_doc_retrieval(in_data):
  
  results = []
  failed = []
  
  for d in tqdm(in_data):
    d_accuracy = []
    predicted_docs = d["predicted_pages"]
    evidence_docs = []
    for i, evidence_set in enumerate(d["evidence"]):
      d_accuracy.append(0)
      evidence_docs = [e[2] for e in evidence_set]
      if not any(evidence_docs):
        # Set 100% for claims without evidence, e.g. NEI
        d_accuracy[i] = 1
        break
        
      for doc in evidence_docs:
        if doc in predicted_docs:
          d_accuracy[i] += 1
      d_accuracy[i] /= len(evidence_docs)
    
    res_obj = {
      "id": d["id"],
      "verifiable": d["verifiable"],
      "claim": d["claim"],
      "evidence_docs": evidence_docs,
      "predicted_docs": predicted_docs,
      "accuracy": max(d_accuracy)
    }
    results.append(res_obj)
    
    if max(d_accuracy) < 1:
      failed.append(res_obj)
    
  avg_accuracy = sum([r["accuracy"] for r in results]) / len(results)

  return {
      "avg_accuracy": avg_accuracy,
      "failed": failed,
      # "results": results
    }
    
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--in-file', type=str, help="results file to be evaluated")
  parser.add_argument('--out-file', type=str, help="file to store the results")
  parser.add_argument('--type', type=str, help="what is to be evaluated e.g. document retrieval?")
  parser.add_argument('--parallel', type=bool, default=True)
  args = parser.parse_args()
  
  in_data = util_funcs.load_jsonl(args.in_file)
  
  if args.type == EvalTypes.DocRetrieval.value:    
    output = eval_doc_retrieval(in_data)
    logger.info(f"Document retrieval accuracy: {output['avg_accuracy']}")
    util_funcs.store_json(output, args.out_file, indent=2)
    logger.info(f"Stored output in '{args.out_file}'")
  else:
    raise argparse.ArgumentError(None, "Invalid '--type' argument")
