import random, itertools
from utils_package import util_funcs
from numpy import random

from src.data.doc_db import DocDB


class Dataset():
  def __init__(self, db_path, batch_size=32) -> None:
    self.train_data = None
    self.dev_data = None
    self.test_data = None
    self.db = DocDB(db_path=db_path)
    self.batch_size = batch_size
  

  def load_train_set(self, file_name):
    if ".jsonl" in file_name:
      self.train_data = util_funcs.load_jsonl(file_name)
    elif ".json" in file_name:
      self.train_data = util_funcs.load_json(file_name)
  

  def load_dev_set(self, file_name):
    if ".jsonl" in file_name:
      self.dev_data = util_funcs.load_jsonl(file_name)
    elif ".json" in file_name:
      self.dev_data = util_funcs.load_json(file_name)
    

  def load_test_set(self, file_name):
    if ".jsonl" in file_name:
      self.test_data = util_funcs.load_jsonl(file_name)
    elif ".json" in file_name:
      self.test_data = util_funcs.load_json(file_name)


  def get_random_samples(self, k):
    if self.train_data:
      rand_idx = random.randint(len(self.train_data), size=(k))
      return [self.train_data[idx] for idx in rand_idx]
    
    if self.dev_data:
      rand_idx = random.randint(len(self.dev_data), size=(k))
      return [self.dev_data[idx] for idx in rand_idx]
      

  def train_data_generator(self):
    pass
  
        
  def get_dev_data_batch(self):
    yield itertools.islice(self.dev_data_generator, self.batch_size)

      