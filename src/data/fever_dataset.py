import pprint, random

import torch
from src.data.dataset import BaseDataset
from typing import List, TypedDict
from transformers import DebertaV2Tokenizer

from src.utils.helpers import create_input_str

pp = pprint.PrettyPrinter(indent=2)

# FEVER to 'deberta-v2-xlarge-mnli' mapping
label2id = {
  "REFUTES": 0,
  "NOT ENOUGH INFO": 1,
  "SUPPORTS": 2
}


class FeverDataSample(TypedDict):
  id: int
  claim: str
  label: str
  verifiable: str
  evidence: List[List[List[str]]]
  

class FEVERDataset(BaseDataset):
  
  def __init__(self, data_file, db_path, tokenizer) -> None:
    super().__init__(data_file, db_path, tokenizer)
      

  def __getitem__(self, idx):
    d = self.data[idx]
    evidence_texts = self.get_evidence_texts(d)
    input_str = create_input_str(d["claim"], evidence_texts)
    inputs = self.tokenizer(input_str, return_tensors="pt", padding="max_length", truncation=True)
    for key in inputs:
      inputs[key] = torch.squeeze(inputs[key])
    label_idx = label2id[d["label"]]
    labels = torch.tensor([label_idx])#.unsqueeze(0)
    return inputs, labels


  def get_sample_by_id(self, id):
    if self.data:
      sample = next((d for d in self.train_data if d["id"] == id), None)
      if sample:
        return sample
            
    return None


  def get_evidence_texts(self, d: FeverDataSample):
    evidence_texts = []
    
    for evidence_set in d["evidence"]:
      evidence_set_texts = []
      
      for evidence in evidence_set:
        doc_id = evidence[2]
        sent_id = evidence[3]
        
        if not doc_id: 
          break
        
        doc_lines_text = self.db.get_doc_lines(doc_id)
        doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
            doc_lines_text.split("\n")]
        
        evidence_set_texts.append([doc_id, doc_lines[sent_id]])

      evidence_texts.append(evidence_set_texts)
      
    return evidence_texts
    

  def get_random_samples_with_text(self, k):
    random_samples = self.get_random_samples(k)
    for d in random_samples:
      d["evidence_texts"] = self.get_evidence_texts(d)
    return random_samples

        
  def dev_data_generator(self):
    random.shuffle(self.dev_data)
    for d in self.dev_data:
      d["evidence_texts"] = self.get_evidence_texts(d)
      yield d

      
      
if __name__ == "__main__":
  
  data_file = "data/fever/dev.jsonl"
  db_path = "data/fever/fever.db"
  model_name = "microsoft/deberta-v2-xlarge-mnli"
  tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
  dataset = FEVERDataset(db_path)
  
  dataset.load_dev_set("data/fever/dev.jsonl")
  random_samples = dataset.get_random_samples_with_text(5)
  
  pp.pprint(random_samples)
  
  
      
    