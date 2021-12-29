from dataset import Dataset
import pprint

pp = pprint.PrettyPrinter(indent=2)

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


  def get_random_samples_with_text(self, k):
    random_samples = self.get_random_samples(k)
    
    # test_sample = self.get_sample_by_id(57842)
    # random_samples.insert(0, test_sample)
    
    for d in random_samples:
      d["evidence_texts"] = []
      
      for evidence_set in d["evidence"]:
        evidence_set_texts = []
        
        for evidence in evidence_set:
          doc_id = evidence[2]
          sent_id = evidence[3]
          
          if not doc_id: 
            break
          
          # doc_text = self.db.get_doc_text(doc_id)
          # doc_sents = doc_text.split(" . ")

          doc_lines_text = self.db.get_doc_lines(doc_id)
          doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
              doc_lines_text.split("\n")]
          
          evidence_set_texts.append([doc_id, doc_lines[sent_id]])

        d["evidence_texts"].append(evidence_set_texts)

    return random_samples
        
    
      
      
if __name__ == "__main__":
  
  db_path = "data/fever/fever.db"
  dataset = FEVERDataset(db_path)
  
  dataset.load_dev_set("data/fever/dev.jsonl")
  random_samples = dataset.get_random_samples_with_text(5)
  
  pp.pprint(random_samples)
  
  
      
    