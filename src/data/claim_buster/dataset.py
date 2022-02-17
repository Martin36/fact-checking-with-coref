import pandas as pd
import torch
from torch.utils.data import Dataset


class ClaimBusterDataset(Dataset):
  def __init__(self, data_file, tokenizer=None) -> None:
    self.data = pd.read_csv(data_file)
    self.tokenizer = tokenizer
    
  def __getitem__(self, idx):
    d = self.data.iloc[idx]
    text = d["Text"]
    verdict = d["Verdict"]
    label = self.convert_verdict_to_label(verdict)
    inputs = self.tokenizer(text, return_tensors="pt", 
                            padding="max_length", truncation=True)
    for key in inputs:
      inputs[key] = torch.squeeze(inputs[key])
    inputs["labels"] = label
    return inputs
  
  def __len__(self):
    return len(self.data)
    
  def convert_verdict_to_label(self, verdict):
    """ verdict -1 or 0 is converted into a non-checkworthy label """
    if verdict < 1:
      return 0
    return 1
  
if __name__ == "__main__":
  
  data_file = "data/claim_buster/crowdsourced.csv"
  ds = ClaimBusterDataset(data_file)
  print(ds[0])