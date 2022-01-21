import torch, utils_package

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaTokenizer, DebertaForSequenceClassification
from transformers import AdamW

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd.grad_mode import no_grad
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from utils_package.util_funcs import store_json, load_jsonl

from src.data.fever_dataset import FEVERDataset
from src.utils.helpers import calc_accuracy, create_dirs_if_not_exist, tensor_dict_to_device
from src.utils.constants import FEVER_ID_2_LABEL, FEVER_LABEL_2_ID

logger = utils_package.logger.get_logger()

class ClaimVerifier():
  
  def __init__(self, model, device, use_gradient_checkpointing=True,
               show_loss_after_steps=10, save_model_after_steps=10000,
               model_save_dir=None, show_eval_loss_after_steps=10000) -> None:
    self.model = model
    self.device = device
    self.model.to(device)    
    self.use_gradient_checkpointing = use_gradient_checkpointing
    self.show_loss_after_steps = show_loss_after_steps
    self.show_eval_loss_after_steps = show_eval_loss_after_steps
    self.save_model_after_steps = save_model_after_steps
    self.model_save_dir = model_save_dir
    

  def train(self, train_dataset, dev_dataset, batch_size=1):
    optimizer = AdamW(self.model.parameters(), lr=1e-5)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    
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
      
      if (step+1) % self.show_eval_loss_after_steps == 0:
        self.calculate_eval_loss(dev_dataloader, step)
      
      
      if (step+1) % self.save_model_after_steps == 0:
        version = (step+1) // self.save_model_after_steps
        self.save_model(version)
      
        
  def predict(self, inputs):
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
    
  
  def calculate_eval_loss(self, dev_dataloader, step):
    avg_loss = 0
    with no_grad():      
      for batch in tqdm(dev_dataloader):
        for key in batch:
          batch[key] = batch[key].to(self.device)
        output = self.model(**batch)
        loss = output.loss
        avg_loss += loss.item()
    avg_loss /= len(dev_dataloader)
    print(f"Loss on dev set after {step+1} steps: {avg_loss}")

  
  def save_model(self, version):
    model_save_name = f"FEVER_entailment_model_v{version}"
    save_path = self.model_save_dir + model_save_name
    self.model.save_pretrained(save_path)
    # torch.save(self.model.state_dict(), save_path)
    print(f"Saved model version {version} in '{save_path}'")
    
