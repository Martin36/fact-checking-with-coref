
from collections import defaultdict
from typing import List


stats = defaultdict(int)

def encode_fever_text(input: str):
  output = input.replace('( ', '-LRB-')
  output = output.replace(' )', '-RRB-')
  output = output.replace(' - ', '-')
  output = output.replace(' :', '-COLON-')
  output = output.replace(' ,', ',')
  output = output.replace(" 's", "'s")
  output = output.replace(' ', '_')
  return output


def decode_fever_text(input: str):
  output = input.replace('-LRB-', '( ')
  output = output.replace('-RRB-', ' )')
  output = output.replace('-', ' - ')
  output = output.replace('-COLON-', ' :')
  output = output.replace(',', ' ,')
  output = output.replace("'s", " 's")
  output = output.replace('_', ' ')
  return output


def create_input_str(claim: str, evidence_texts: List[str]):
  result = f"{claim} "
  if len(evidence_texts) > 1:
    stats["multiple_evidence"] += 1
  evidence_texts = evidence_texts[0]  # TODO: What if there is multiple evidence sets?
  evidence_str_list = [f"{decode_fever_text(evi[0])} {evi[1]}" 
                      for evi in evidence_texts]
  evidence_concat = " [SEP] ".join(evidence_str_list)
  result += f"[SEP] {evidence_concat}"
  return result    
    

def tensor_dict_to_device(dict, device):
  for key in dict:
    dict[key] = dict[key].to(device)
  return dict


def calc_accuracy(pred_labels, gold_labels):
  accuracy = 0
  for pred, gold in zip(pred_labels, gold_labels):
    if pred == gold:
      accuracy += 1
  accuracy /= len(pred_labels)
  return accuracy


def get_fever_doc_lines(doc_text):
  return [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
            doc_text.split("\n")]
