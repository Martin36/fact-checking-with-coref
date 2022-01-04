from src.data.fever_dataset import FEVERDataset
from src.utils.helpers import decode_fever_text
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch

class ClaimVerifier():
  
  def __init__(self, dataset, model_name) -> None:
    self.dataset = dataset
    self.model_name = model_name
    self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    self.model = DebertaV2ForSequenceClassification.from_pretrained(model_name)
    
  
  def create_input_str(self, claim, evidence_texts):
    result = f"[CLS] {claim} "
    if len(evidence_texts) > 1:
      print("Multiple evidence ")
    evidence_texts = evidence_texts[0]  # TODO: What if there is multiple evidence sets?
    evidence_str_list = [f"{decode_fever_text(evi[0])} {evi[1]}" 
                       for evi in evidence_texts]
    evidence_concat = " [SEP] ".join(evidence_str_list)
    result += f"[SEP] {evidence_concat}"
    return result    
    
  def predict(self, claim, evidence):
    inputs = self.tokenizer(claim, return_tensors="pt")
    pass
  
  def calc_dev_acc(self):
    pass
    
    
if __name__ == "__main__":
  
  db_path = "data/fever/fever.db"
  dev_data_path = "data/fever/dev.jsonl"
  
  fever_ds = FEVERDataset(db_path)
  fever_ds.load_dev_set(dev_data_path)
  
  # TODO: Use 'document-level-FEVER' model instead
  model_name = "microsoft/deberta-v2-xlarge"
  
  claim_verifier = ClaimVerifier(fever_ds, model_name)
  
  for d in fever_ds.dev_data_generator():
    input_str = claim_verifier.create_input_str(d["claim"], d["evidence_texts"])
    print(input_str)
  
  # for batch in fever_ds.get_dev_data_batch():
    
  
  
  
  
  
  