import pprint, random, torch, utils_package
import numpy as np

from src.data.dataset import BaseDataset
from tqdm import tqdm

from src.utils.helpers import (
  create_input_str, 
  get_fever_doc_lines, 
  get_non_empty_indices, 
  get_random_from_list, 
  get_evidence_texts
)
from src.utils.constants import FEVER_LABEL_2_ID

pp = pprint.PrettyPrinter(indent=2)
logger = utils_package.logger.get_logger()


class FEVERDataset(BaseDataset):
  
  def __init__(self, data_file, db_path=None, tokenizer=None) -> None:
    super().__init__(data_file, tokenizer, db_path=db_path)
    
    if "evidence_texts" in self.data[0]:
      self.has_evidence_texts = True
    else:
      self.has_evidence_texts = False      
    if "label" in self.data[0]:
      self.has_labels = True
    else:
      self.has_labels = False


  def __getitem__(self, idx):
    d = self.data[idx]
    if self.has_evidence_texts:
      evidence_texts = d["evidence_texts"]
    else:
      evidence_texts = get_evidence_texts(self.db, d)
    input_str = create_input_str(d["claim"], evidence_texts)
    inputs = self.tokenizer(input_str, return_tensors="pt", padding="max_length", truncation=True)
    for key in inputs:
      inputs[key] = torch.squeeze(inputs[key])
    if self.has_labels:
      label_idx = FEVER_LABEL_2_ID[d["label"]]
      # labels = torch.tensor([label_idx])#.unsqueeze(0)
      # inputs["labels"] = labels
      inputs["labels"] = label_idx
    return inputs


  def get_sample_at_index(self, idx):
    return self.data[idx]


  def get_random_samples_with_text(self, k):
    random_samples = self.get_random_samples(k)
    for d in random_samples:
      d["evidence_texts"] = get_evidence_texts(self.db, d)
    return random_samples

        
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
  
  data_file = "data/fever/doc_retrieval/dev.wiki7.jsonl"
  db_path = "data/fever/fever.db"

  dataset = FEVERDataset(data_file, db_path)
    
  out_file = "data/fever/dev_with_evidence.jsonl"
  dataset.create_ds_with_evidence_texts(out_file, sample_nei=True)
      
    