import pprint, random, torch, utils_package

from src.data.dataset import BaseDataset
from tqdm import tqdm

from src.utils.helpers import create_input_str, get_fever_doc_lines
from src.utils.types import FeverDataSample
from src.utils.constants import label2id

pp = pprint.PrettyPrinter(indent=2)
logger = utils_package.logger.get_logger()


class FEVERDataset(BaseDataset):
  
  def __init__(self, data_file, db_path, tokenizer=None) -> None:
    super().__init__(data_file, db_path, tokenizer)
      

  def __getitem__(self, idx):
    d = self.data[idx]
    evidence_texts = self.get_evidence_texts(d)
    input_str = create_input_str(d["claim"], evidence_texts)
    inputs = self.tokenizer(input_str, return_tensors="pt", padding="max_length", truncation=True)
    for key in inputs:
      inputs[key] = torch.squeeze(inputs[key])
    label_idx = label2id[d["label"]]
    labels = torch.tensor([label_idx])#.unsqueeze(0)
    return inputs, labels


  def get_sample_by_id(self, id):
    if self.data:
      sample = next((d for d in self.train_data if d["id"] == id), None)
      if sample:
        return sample
            
    return None


  def get_evidence_texts(self, d: FeverDataSample):
    evidence_texts = []
    
    for evidence_set in d["evidence"]:
      evidence_set_texts = []
      
      for evidence in evidence_set:
        doc_id = evidence[2]
        sent_id = evidence[3]
        
        if not doc_id: 
          break
        
        doc_lines_text = self.db.get_doc_lines(doc_id)
        doc_lines = get_fever_doc_lines(doc_lines_text)
        
        evidence_set_texts.append([doc_id, doc_lines[sent_id]])

      evidence_texts.append(evidence_set_texts)
      
    return evidence_texts
    

  def get_random_samples_with_text(self, k):
    random_samples = self.get_random_samples(k)
    for d in random_samples:
      d["evidence_texts"] = self.get_evidence_texts(d)
    return random_samples

        
  def dev_data_generator(self):
    random.shuffle(self.data)
    for d in self.data:
      d["evidence_texts"] = self.get_evidence_texts(d)
      yield d

  
  def create_ds_with_evidence_texts(self, out_file):
    for d in tqdm(self.data):
      d["evidence_texts"] = self.get_evidence_texts(d)
    utils_package.store_jsonl(self.data, out_file)
    logger.info(f"Stored dataset with evidence in '{out_file}'")
    
      
if __name__ == "__main__":
  
  data_file = "data/fever/dev.jsonl"
  db_path = "data/fever/fever.db"

  dataset = FEVERDataset(data_file, db_path)
  
  random_samples = dataset.get_random_samples_with_text(5)
  
  pp.pprint(random_samples)
  
  out_file = "data/fever/dev_with_evidence.jsonl"
  dataset.create_ds_with_evidence_texts(out_file)
      
    