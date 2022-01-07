import random, itertools
from utils_package import util_funcs
from numpy import random
from torch.utils.data import Dataset
from src.data.doc_db import DocDB

# TODO: is it possible to make this abstract?
class BaseDataset(Dataset):
  def __init__(self, data_file, db_path, tokenizer, batch_size=32) -> None:
    self.data = self.load_data(data_file)
    self.db = DocDB(db_path=db_path)
    self.tokenizer = tokenizer
    self.batch_size = batch_size
  

  def __len__(self):
    return len(self.data)
  
  
  def __getitem__(self):
    pass
  

  def load_data(self, data_file):
    if ".jsonl" in data_file:
      return util_funcs.load_jsonl(data_file)
    elif ".json" in data_file:
      return util_funcs.load_json(data_file)
  

  def get_random_samples(self, k):
    if self.data:
      rand_idx = random.randint(len(self.data), size=(k))
      return [self.data[idx] for idx in rand_idx]      


  def train_data_generator(self):
    pass
  
        
  def get_dev_data_batch(self):
    yield itertools.islice(self.dev_data_generator, self.batch_size)

      