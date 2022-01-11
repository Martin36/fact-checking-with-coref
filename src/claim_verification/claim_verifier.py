import numpy as np
import torch, utils_package

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaTokenizer, DebertaForSequenceClassification
from transformers import TrainingArguments, Trainer, AdamW

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from src.data.fever_dataset import FEVERDataset
from src.utils.helpers import calc_accuracy, tensor_dict_to_device
from src.utils.constants import label2id

logger = utils_package.logger.get_logger()


def compute_metrics(p):
  pred, labels = p
  pred = np.argmax(pred, axis=1)

  accuracy = accuracy_score(y_true=labels, y_pred=pred)
  recall = recall_score(y_true=labels, y_pred=pred)
  precision = precision_score(y_true=labels, y_pred=pred)
  f1 = f1_score(y_true=labels, y_pred=pred)

  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


class ClaimVerifier():
  
  def __init__(self, model, device, use_gradient_checkpointing=True,
               show_loss_after_steps=10) -> None:
    self.model = model
    self.device = device
    self.model.to(device)    
    self.use_gradient_checkpointing = use_gradient_checkpointing
    self.show_loss_after_steps = show_loss_after_steps
    

  def train_with_trainer(self, train_dataset, dev_dataset):
    out_dir = "models/deberta_base_mnli_finetuned"
    training_args = TrainingArguments(
      out_dir, 
      overwrite_output_dir=True, 
      evaluation_strategy="steps", 
      eval_steps=100,
      per_device_train_batch_size=1,
      per_device_eval_batch_size=1,
      learning_rate=5e-5,
      weight_decay=0,
      num_train_epochs=1,
      save_strategy="steps",
      save_steps=500,
      save_total_limit=10,
      gradient_checkpointing=True
    )
    
    trainer = Trainer(
      model=self.model, 
      args=training_args, 
      train_dataset=train_dataset, 
      eval_dataset=dev_dataset,
      compute_metrics=compute_metrics,
    )
    
    trainer.train()


  def train(self, train_dataset, dev_dataset):
    optimizer = AdamW(self.model.parameters(), lr=1e-5)
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    
    if self.use_gradient_checkpointing:
      self.model.gradient_checkpointing_enable()
    
    train_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
      for key in batch:
        batch[key] = batch[key].to(self.device)
      optimizer.zero_grad()
      output = self.model(**batch)
      loss = output.loss
      loss.backward()
      optimizer.step()
      train_loss += loss.item()  
      
      if (step+1) % self.show_loss_after_steps == 0:
        print(f"Train loss after {step+1} steps: {train_loss/(step+1)}") 
      
        
  def predict(self, inputs, labels):
    with torch.no_grad():
      inputs = tensor_dict_to_device(inputs, self.device)
      # labels = torch.squeeze(labels)
      # labels = labels.to(self.device)
      outputs = self.model(**inputs)
      logits = outputs.logits
      return logits


  def convert_logits_to_labels(self, logits):
    _, idxs = torch.max(logits, dim=1)
    return idxs.tolist()
    
  
  def save_model(self, path):
    pass
    

def predict(claim_verifier, dataset):
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
  pred_labels = []
  gold_labels = []
  for inputs, labels in tqdm(dataloader):
    logits = claim_verifier.predict(inputs, labels)    
    pred_labels += claim_verifier.convert_logits_to_labels(logits)
    gold_labels += torch.squeeze(labels).tolist()
    
  accuracy = calc_accuracy(pred_labels, gold_labels)
  logger.info(f"Accuracy for model '{model_name}' on dev set is: {accuracy}")

  target_names = list(label2id.keys())
  cls_report = classification_report(gold_labels, pred_labels, target_names=target_names)
  print(cls_report)

    
if __name__ == "__main__":
  
  dev_data_path = "data/fever/dev.jsonl"
  train_data_path = "data/fever/train_with_evidence.jsonl"
  # model_name = "microsoft/deberta-v2-xlarge"
  # model_name = "models/document-level-FEVER/RTE-debertav2-MNLI"
  # model_name = "microsoft/deberta-v2-xlarge-mnli"
  model_name = "microsoft/deberta-base-mnli"
  model = DebertaForSequenceClassification.from_pretrained(model_name)
  # tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
  tokenizer = DebertaTokenizer.from_pretrained(model_name)

  batch_size = 8
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dev_dataset = FEVERDataset(dev_data_path, tokenizer=tokenizer)
  train_dataset = FEVERDataset(train_data_path, tokenizer=tokenizer)

  claim_verifier = ClaimVerifier(model, device)
  
  # predict(claim_verifier, dev_dataset)
  
  claim_verifier.train(train_dataset, dev_dataset)
  
  
  
  
  
  