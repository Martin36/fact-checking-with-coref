from src.data.fever_dataset import FEVERDataset
from src.utils.helpers import create_input_str
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

LABEL_2_IDX = {
  "SUPPORTS": 0, 
  "REFUTES": 1, 
  "NOT ENOUGH INFO": 2
}

IDX_2_LABEL = {
  0: "SUPPORTS", 
  1: "REFUTES", 
  2: "NOT ENOUGH INFO"
}

# For 'deberta-v2-xlarge-mnli'
id2label = {
  0: "CONTRADICTION",
  1: "NEUTRAL",
  2: "ENTAILMENT"
}

# FEVER to 'deberta-v2-xlarge-mnli' mapping
label2id = {
  "REFUTES": 0,
  "NOT ENOUGH INFO": 1,
  "SUPPORTS": 2
}


class ClaimVerifier():
  
  def __init__(self, model_name, use_gpu) -> None:
    self.model = DebertaV2ForSequenceClassification.from_pretrained(model_name)
    self.use_gpu = use_gpu
    if use_gpu:
      self.model.to_gpu()
    
    
  def predict(self, claim, evidence_texts, label_text):
    input_str = create_input_str(claim, evidence_texts)
    inputs = self.tokenizer(input_str, return_tensors="pt")
    label_idx = label2id[label_text]
    labels = torch.tensor([label_idx])#.unsqueeze(0)
    outputs = self.model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    return loss, logits


  def predict_batch(self, inputs, labels):
    with torch.no_grad():
      labels = torch.squeeze(labels)
      outputs = self.model(**inputs, labels=labels)
      logits = outputs.logits
      return logits


  def convert_logits_to_labels(self, logits):
    out, idxs = torch.max(logits, dim=1)
    return idxs.tolist()
  
  def calc_accuracy(self, pred_labels, gold_labels):
    accuracy = 0
    for pred, gold in zip(pred_labels, gold_labels):
      if pred == gold:
        accuracy += 1
    accuracy /= len(pred_labels)
    return accuracy
  
  def save_model(self, path):
    pass
    
    
if __name__ == "__main__":
  
  db_path = "data/fever/fever.db"
  dev_data_path = "data/fever/dev.jsonl"
  # model_name = "microsoft/deberta-v2-xlarge"
  # model_name = "models/document-level-FEVER/RTE-debertav2-MNLI"
  model_name = "microsoft/deberta-v2-xlarge-mnli"

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

  fever_ds = FEVERDataset(dev_data_path, db_path, tokenizer)
  
  dev_dataloader = DataLoader(fever_ds, batch_size=4, shuffle=True)
  
  claim_verifier = ClaimVerifier(model_name, device)
  
  pred_labels = []
  gold_labels = []
  for inputs, labels in tqdm(dev_dataloader):    
    logits = claim_verifier.predict_batch(inputs, labels)    
    pred_labels += claim_verifier.convert_logits_to_labels(logits)
    gold_labels += torch.squeeze(labels).tolist()
    

  for d in fever_ds.dev_data_generator():
    if d["verifiable"] == "NOT VERIFIABLE":
      continue
    input_str = claim_verifier.create_input_str(d["claim"], d["evidence_texts"])
    print(input_str)
  
  # for batch in fever_ds.get_dev_data_batch():
    
  
  
  
  
  
  