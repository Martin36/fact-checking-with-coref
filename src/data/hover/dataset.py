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
        
        [doc_id, sent_id] = evidence
          
        if not doc_id:
          break
          
        doc_text = self.db.get_doc_lines(doc_id)
        doc_sents = doc_text.split("\n")

        # doc_lines_text = self.db.get_doc_lines(doc_id)
        # doc_lines = [doc_line.split("\t")[1] if len(doc_line.split("\t")[1]) > 1 else "" for doc_line in
        #     doc_lines_text.split("\n")]
          
        d["evidence_texts"].append([doc_id, doc_sents[sent_id]])

    return random_samples


def concat_evidence_texts(evidence_texts, with_title=False):
  result = ""
  for evidence in evidence_texts:
    if with_title:
      result += f"({evidence[0]}) {evidence[1]} "
    else:
      result += evidence[1] + " "
  return result.strip()


if __name__ == "__main__":
  db_path = "data/hover/wiki_with_lines.db"
  dataset = HoVerDataset(db_path)
  
  dataset.load_dev_set("data/hover/dev.json")
  random_samples = dataset.get_random_samples_with_text(5)
  
  for d in random_samples:
    print("\n")
    print(f"Label: {d['label']}")
    print(f"Claim: {d['claim']}")
    print(f"Evidence: {concat_evidence_texts(d['evidence_texts'], with_title=True)}")
      
  
  

  