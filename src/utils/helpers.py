import os
import random
from collections import defaultdict
from typing import List
from utils_package.logger import get_logger

from src.data.doc_db import DocDB

logger = get_logger()

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


# def get_fever_doc_lines(doc_text):
#   return [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
#             doc_text.split("\n")]


def get_fever_doc_lines(doc_text):
  result = []
  doc_lines = doc_text.split("\n")
  for doc_line in doc_lines:
    if len(doc_line) == 0:
      continue
    doc_line_split = doc_line.split("\t")
    if len(doc_line_split[1]) > 1:
      result.append(doc_line_split[1])
    else:
      result.append("")
  return result



def get_evidence_pages(evidence_sents: List[List[str]]):
  doc_titles = set()
  for evi in evidence_sents:
    doc_titles.add(evi[2])
  return list(doc_titles)


def get_random_from_list(list, n=None, max_n=None):
  """Gets n random element from the provided list
  
  Args:
      list (list): The list to pick elements from
      
      n (int, optional): Number of elements to pick. 
      If not provided, it will pick a random nr of samples from the list. 
      Defaults to None.
      
      max_n (int, optional): Max number of elements to pick. 
      Prove this if you still want the number of elements to be random,
      but want to limit the size of the returned list. If n is provided
      it will override this
      Defaults to None.

  Returns:
      list: List of randomly picked elements
  """
  
  if not list or len(list) == 0:
    return []
  if not n:
    n = random.randint(1, max_n if max_n else len(list))
  return random.sample(list, n if n <= len(list) else len(list))


def get_non_empty_indices(list: List[str]):
  return [i for i, txt in enumerate(list) if len(txt) > 0]


def get_evidence_texts(db: DocDB, d: dict):
  """Gets the actual text of the evidence

  Args:
      db (DocDB): The database where the documents are located
      d (dict): A dict that has "evidence" as one of its properties. 
      Evidence should be on the form [_, _, doc_title, sentence_id]

  Returns:
      List[str]: A list of the texts of the provided evidence sentences
  """

  evidence_texts = []
  
  for evidence_set in d["evidence"]:
    evidence_set_texts = []
    
    for evidence in evidence_set:
      if len(evidence) == 0:
        continue
      doc_id = evidence[2]
      sent_id = evidence[3]
      
      if not doc_id: 
        break
      
      doc_lines_text = db.get_doc_lines(doc_id)
      doc_lines = get_fever_doc_lines(doc_lines_text)
      
      evidence_set_texts.append([doc_id, doc_lines[sent_id]])

    evidence_texts.append(evidence_set_texts)
    
  return evidence_texts
  

def filter_empty(list: List[str]):
  return [elem for elem in list if len(elem) > 0]