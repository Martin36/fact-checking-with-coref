import json
from dataset import Dataset
import pprint, bz2

pp = pprint.PrettyPrinter(indent=2)

class HoVerDataset(Dataset):
  
  def __init__(self, db_path) -> None:
    super().__init__(db_path)
      
      
  def get_random_samples_with_text(self, k):
    random_samples = self.get_random_samples(k)
        
    for d in random_samples:
      d["evidence_texts"] = []
      
      for evidence in d["supporting_facts"]:
        
        doc_id = evidence[0]
        sent_id = evidence[1]
          
        if not doc_id: 
          break
          
        doc_text = self.db.get_doc_text(doc_id)
        doc_sents = doc_text.split(" . ")

        # doc_lines_text = self.db.get_doc_lines(doc_id)
        # doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
        #     doc_lines_text.split("\n")]
          
        d["evidence_texts"].append([doc_id, doc_sents[sent_id]])

    return random_samples

    
if __name__ == "__main__":
  db_path = "data/hover/wiki_wo_links.db"
  dataset = HoVerDataset(db_path)
  
  dataset.load_dev_set("data/hover/dev.json")
  random_samples = dataset.get_random_samples(5)
  
  pp.pprint(random_samples)
  
  

  