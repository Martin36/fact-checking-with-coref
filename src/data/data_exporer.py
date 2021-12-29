from utils_package.util_funcs import load_jsonl
from numpy import random
import pprint

pp = pprint.PrettyPrinter(indent=2)

class DataExplorer():
  
  def __init__(self) -> None:
    self.data = None
    pass
  
  def import_data(self, file_name):
    if ".jsonl" in file_name:
      self.data = load_jsonl(file_name)
    
    
  def get_random_samples(self, k):
    rand_idx = random.randint(len(self.data), size=(5))
    return [self.data[idx] for idx in rand_idx]
    

if __name__ == "__main__":
  data_explorer = DataExplorer()
  data_explorer.import_data("data/fever/dev.p7.s5.jsonl")
  rand_samples = data_explorer.get_random_samples(5)
  
  pp.pprint(rand_samples)
  
  
  