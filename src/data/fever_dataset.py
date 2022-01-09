import pprint, random, torch, utils_package
import numpy as np

from src.data.dataset import BaseDataset
from tqdm import tqdm

from src.utils.helpers import create_input_str, get_fever_doc_lines, get_non_empty_indices, get_random_from_list
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


  def get_sample_evidence_sents(self, pages, max_n=5):
    all_sent_id_pairs = []
    for page in pages:
      doc_text = self.db.get_doc_lines(page)
      if not doc_text:
        continue
      doc_lines = get_fever_doc_lines(doc_text)
      sentence_ids = np.arange(len(doc_lines)).tolist()
      page_name_list = [page] * len(doc_lines)
      sent_id_page_pairs = zip(doc_lines, sentence_ids, page_name_list)
      non_empty_sents_indices = get_non_empty_indices(doc_lines)
      sent_id_pairs = [[sent, sent_id, page] 
                       for sent, sent_id, page in sent_id_page_pairs 
                       if sent_id in non_empty_sents_indices]
      all_sent_id_pairs += sent_id_pairs

    return get_random_from_list(all_sent_id_pairs, max_n=max_n)
  
  
  def extract_evidence_from_sample_evidence(self, sample_evidence):
    return [[[None, None, doc_id, sent_id] 
             for _, sent_id, doc_id in sample_evidence]]

  def extract_evidence_texts_from_sample_evidence(self, sample_evidence):
    return [[[doc_id, sent_text] 
             for sent_text, _, doc_id in sample_evidence]]
  

  def create_ds_with_evidence_texts(self, out_file, sample_nei=False):      
    
    for d in tqdm(self.data):
      if sample_nei and d["verifiable"] == "NOT VERIFIABLE":
        pages = d["predicted_pages"] if len(d["predicted_pages"]) > 0 else d["wiki_results"]
        sample_evidence = self.get_sample_evidence_sents(pages)
        d["evidence"] = self.extract_evidence_from_sample_evidence(sample_evidence)
        d["evidence_texts"] = self.extract_evidence_texts_from_sample_evidence(sample_evidence)
      else:        
        d["evidence_texts"] = self.get_evidence_texts(d)
    
    utils_package.store_jsonl(self.data, out_file)
    logger.info(f"Stored dataset with evidence in '{out_file}'")
    
      
if __name__ == "__main__":
  
  data_file = "data/fever/doc_retrieval/train.wiki7.jsonl"
  db_path = "data/fever/fever.db"

  dataset = FEVERDataset(data_file, db_path)
    
  out_file = "data/fever/train_with_evidence.jsonl"
  dataset.create_ds_with_evidence_texts(out_file, sample_nei=True)
      
    