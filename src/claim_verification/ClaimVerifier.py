import torch, utils_package
from src.data.fever_dataset import FEVERDataset
from src.utils.helpers import calc_accuracy, create_input_str, tensor_dict_to_device
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaTokenizer, DebertaForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = utils_package.logger.get_logger()

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
  
  def __init__(self, model_name, device) -> None:
    # self.model = DebertaV2ForSequenceClassification.from_pretrained(model_name)
    self.model = DebertaForSequenceClassification.from_pretrained(model_name)
    self.device = device
    self.model.to(device)    
    
    
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
      inputs = tensor_dict_to_device(inputs, self.device)
      labels = torch.squeeze(labels)
      labels = labels.to(self.device)
      outputs = self.model(**inputs, labels=labels)
      logits = outputs.logits
      return logits


  def convert_logits_to_labels(self, logits):
    out, idxs = torch.max(logits, dim=1)
    return idxs.tolist()
    
  
  def save_model(self, path):
    pass
    
    
if __name__ == "__main__":
  
  db_path = "data/fever/fever.db"
  dev_data_path = "data/fever/dev.jsonl"
  # model_name = "microsoft/deberta-v2-xlarge"
  # model_name = "models/document-level-FEVER/RTE-debertav2-MNLI"
  # model_name = "microsoft/deberta-v2-xlarge-mnli"
  model_name = "microsoft/deberta-base-mnli"
  batch_size = 8
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
  tokenizer = DebertaTokenizer.from_pretrained(model_name)

  fever_ds = FEVERDataset(dev_data_path, db_path, tokenizer)
  
  dev_dataloader = DataLoader(fever_ds, batch_size=batch_size, shuffle=True)
  
  claim_verifier = ClaimVerifier(model_name, device)
  
  pred_labels = []
  gold_labels = []
  for inputs, labels in tqdm(dev_dataloader):    
    logits = claim_verifier.predict_batch(inputs, labels)    
    pred_labels += claim_verifier.convert_logits_to_labels(logits)
    gold_labels += torch.squeeze(labels).tolist()
    
  accuracy = calc_accuracy(pred_labels, gold_labels)
  logger.info(f"Accuracy for model '{model_name}' on dev set is: {accuracy}")

    
  
  
  
  
  
  