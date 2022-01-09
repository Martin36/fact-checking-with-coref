import torch, utils_package

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaTokenizer, DebertaForSequenceClassification
from transformers import TrainingArguments, Trainer, AdamW

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

from src.data.fever_dataset import FEVERDataset
from src.utils.helpers import calc_accuracy, tensor_dict_to_device
from src.utils.constants import label2id

logger = utils_package.logger.get_logger()

class ClaimVerifier():
  
  def __init__(self, model_name, device) -> None:
    # self.model = DebertaV2ForSequenceClassification.from_pretrained(model_name)
    self.model = DebertaForSequenceClassification.from_pretrained(model_name)
    self.device = device
    self.model.to(device)    
    
  # TODO: 
  def train(self, train_dataset, dev_dataset):
    out_dir = "models/deberta_base_mnli_finetuned"
    training_args = TrainingArguments(
      out_dir, 
      overwrite_output_dir=True, 
      evaluation_strategy="steps", 
      eval_steps=100,
      per_device_train_batch_size=8,
      per_device_eval_batch_size=8,
      learning_rate=5e-5,
      weight_decay=0,
      num_train_epochs=3,
      save_strategy="steps",
      save_steps=500,
      save_total_limit=10,
    )
    
    trainer = Trainer(
      model=self.model, 
      args=training_args, 
      train_dataset=train_dataset, 
      eval_dataset=dev_dataset
    )
    
    optimizer = AdamW(self.model.parameters(), lr=1e-5)
    
    trainer.train()

        
  def predict(self, inputs, labels):
    with torch.no_grad():
      inputs = tensor_dict_to_device(inputs, self.device)
      labels = torch.squeeze(labels)
      labels = labels.to(self.device)
      outputs = self.model(**inputs, labels=labels)
      logits = outputs.logits
      return logits


  def convert_logits_to_labels(self, logits):
    _, idxs = torch.max(logits, dim=1)
    return idxs.tolist()
    
  
  def save_model(self, path):
    pass
    
    
if __name__ == "__main__":
  
  db_path = "data/fever/fever.db"
  dev_data_path = "data/fever/dev.jsonl"
  train_data_path = "data/fever/train.jsonl"
  # model_name = "microsoft/deberta-v2-xlarge"
  # model_name = "models/document-level-FEVER/RTE-debertav2-MNLI"
  # model_name = "microsoft/deberta-v2-xlarge-mnli"
  model_name = "microsoft/deberta-base-mnli"
  batch_size = 8
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
  tokenizer = DebertaTokenizer.from_pretrained(model_name)

  dev_dataset = FEVERDataset(dev_data_path, db_path, tokenizer)
  train_dataset = FEVERDataset(train_data_path, db_path, tokenizer)
  
  dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
  
  claim_verifier = ClaimVerifier(model_name, device)
  
  pred_labels = []
  gold_labels = []
  for inputs, labels in tqdm(dev_dataloader):    
    logits = claim_verifier.predict(inputs, labels)    
    pred_labels += claim_verifier.convert_logits_to_labels(logits)
    gold_labels += torch.squeeze(labels).tolist()
    
  accuracy = calc_accuracy(pred_labels, gold_labels)
  logger.info(f"Accuracy for model '{model_name}' on dev set is: {accuracy}")

  target_names = list(label2id.keys())
  cls_report = classification_report(gold_labels, pred_labels, target_names=target_names)
  print(cls_report)
    
  
  
  
  
  
  