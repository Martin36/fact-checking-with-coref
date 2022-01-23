import pprint

from src.data.fever_dataset import FEVERDataset

pp = pprint.PrettyPrinter(indent=2)


if __name__ == "__main__":
  
  data_file = "data/fever/dev.jsonl"
  db_path = "data/fever/fever.db"

  dataset = FEVERDataset(data_file, db_path)
    
  random_samples = dataset.get_random_samples_with_text(5)
  
  pp.pprint(random_samples)
      
