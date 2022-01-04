import pprint, random
from src.data.dataset import Dataset
from typing import List, TypedDict

pp = pprint.PrettyPrinter(indent=2)

class FeverDataSample(TypedDict):
  id: int
  claim: str
  label: str
  verifiable: str
  evidence: List[List[List[str]]]
  

class FEVERDataset(Dataset):
  
  def __init__(self, db_path) -> None:
    super().__init__(db_path)
      

  def get_sample_by_id(self, id):
    if self.train_data:
      sample = next((d for d in self.train_data if d["id"] == id), None)
      if sample:
        return sample
      
    if self.dev_data:
      sample = next((d for d in self.dev_data if d["id"] == id), None)
      if sample:
        return sample
      
    if self.test_data:
      sample = next((d for d in self.test_data if d["id"] == id), None)
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
        doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
            doc_lines_text.split("\n")]
        
        evidence_set_texts.append([doc_id, doc_lines[sent_id]])

      evidence_texts.append(evidence_set_texts)
      
    return evidence_texts
    

  def get_random_samples_with_text(self, k):
    random_samples = self.get_random_samples(k)
    for d in random_samples:
      d["evidence_texts"] = self.get_evidence_texts(d)
    return random_samples

        
  def dev_data_generator(self):
    random.shuffle(self.dev_data)
    for d in self.dev_data:
      d["evidence_texts"] = self.get_evidence_texts(d)
      yield d

      
      
if __name__ == "__main__":
  
  db_path = "data/fever/fever.db"
  dataset = FEVERDataset(db_path)
  
  dataset.load_dev_set("data/fever/dev.jsonl")
  random_samples = dataset.get_random_samples_with_text(5)
  
  pp.pprint(random_samples)
  
  
      
    