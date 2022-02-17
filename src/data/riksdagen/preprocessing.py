from collections import defaultdict
import pandas as pd
from utils_package.util_funcs import load_json
from nltk.tokenize import sent_tokenize

stats = defaultdict(int)

def create_sentences_table(data):
  
  result_objs = []
  for statement in data["statements"]:
    paragraphs_sv = statement["text"].split("\n")
    paragraphs_en = statement["text_en"].split("\n")
    
    assert len(paragraphs_en) == len(paragraphs_sv)
    
    for i in range(len(paragraphs_sv)):
      p_sv = paragraphs_sv[i]
      p_en = paragraphs_en[i]
      
      sentences_sv = sent_tokenize(p_sv)
      sentences_en = sent_tokenize(p_en)
      
      if len(sentences_sv) != len(sentences_en):
        stats["paragraphs_with_unequal_nr_of_sentences"] += 1
        len_longest_sent_set = max(len(sentences_sv), len(sentences_en))
        for j in range(len_longest_sent_set):
          sent_sv = None
          sent_en = None
          if j < len(sentences_sv):
            sent_sv = sentences_sv[j]
          if j < len(sentences_en):
            sent_en = sentences_en[j]
          result_obj = {
            "speaker": statement["speaker"],
            "sent_sv": sent_sv,
            "sent_en": sent_en,
            "sent_idx": j,
            "paragraph_idx": i
          }
          result_objs.append(result_obj)
      else:      
        for j in range(len(sentences_sv)):
          sent_sv = sentences_sv[j]
          sent_en = sentences_en[j]
          
          result_obj = {
            "speaker": statement["speaker"],
            "sent_sv": sent_sv,
            "sent_en": sent_en,
            "sent_idx": j,
            "paragraph_idx": i
          }
          
          result_objs.append(result_obj)

  df = pd.DataFrame(result_objs)
  
  out_path = "data/riksdagen/partiledardebatt-22-01-12.csv"
  df.to_csv(out_path)
  
  
def handle_varying_nr_of_sents():
  pass


def extract_numerical_sentences():
  pass


def split_sentences():
  pass
  
  
if __name__ == "__main__":
  data_path = "data/riksdagen/partiledardebatt-12-januari-2022.json"
  data = load_json(data_path)
  create_sentences_table(data)
  
    
