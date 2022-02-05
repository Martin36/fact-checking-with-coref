from tqdm import tqdm
from src.utils.helpers import encode_hover_to_fever_text

from utils_package.util_funcs import load_json, store_jsonl

# source format
  # {
  #   "uid": "042339bf-0374-4ab3-ab49-6df5f12d868e",
  #   "claim": "The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$.",
  #   "supporting_facts": [
  #     [
  #       "Life Goes On (Fergie song)",
  #       2
  #     ],
  #     [
  #       "M.I.L.F. $",
  #       1
  #     ]
  #   ],
  #   "label": "SUPPORTED",
  #   "num_hops": 2,
  #   "hpqa_id": "5abed82a5542993fe9a41d51"
  # },

# dest format
# {
#   "id": 137334, 
#   "verifiable": "VERIFIABLE", 
#   "label": "SUPPORTS", 
#   "claim": "Fox 2000 Pictures released the film Soul Food.", 
#   "evidence": [
#     [
#       [289914, 283015, "Soul_Food_-LRB-film-RRB-", 0]
#     ], 
#     [
#       [291259, 284217, "Soul_Food_-LRB-film-RRB-", 0]
#     ], 
#     [
#       [293412, 285960, "Soul_Food_-LRB-film-RRB-", 0]
#     ], 
#     [
#       [337212, 322620, "Soul_Food_-LRB-film-RRB-", 0]
#     ], 
#     [
#       [337214, 322622, "Soul_Food_-LRB-film-RRB-", 0]
#     ]
#   ]
# }


def convert_hover_data_to_fever_format(data):
  result = []
  for d in tqdm(data):
    fever_d = convert_hover_sample_to_fever_format(d)
    result.append(fever_d)
  return result


def convert_hover_sample_to_fever_format(d):
  
  verifiable = "NOT VERIFIABLE" if d["label"] == "NOT ENOUGH INFO" else "VERIFIABLE"
  
  result = dict()
  result["id"] = d["uid"]
  result["claim"] = d["claim"]
  result["label"] = d["label"]
  result["verifiable"] = verifiable
  result["evidence"] = convert_hover_to_fever_evidence(d)
  
  return result
  

def convert_hover_to_fever_evidence(d):
  evidence_sets = []
  evidence_set = []
  for hover_evidence in d["supporting_facts"]:
    fever_evidence = [None, None, encode_hover_to_fever_text(hover_evidence[0]), hover_evidence[1]]
    evidence_set.append(fever_evidence)
  
  evidence_sets.append(evidence_set)
  return evidence_sets


if __name__ == "__main__":
  hover_path = "data/hover/dev.json"
  out_path = "data/hover/dev_fever_format.jsonl"
  hover_data = load_json(hover_path)
  fever_format_data = convert_hover_data_to_fever_format(hover_data)
  store_jsonl(fever_format_data, out_path)
  print(f"Stored result in '{out_path}'")
  
  