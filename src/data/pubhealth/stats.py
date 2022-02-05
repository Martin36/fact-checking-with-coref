import os
from typing import List
import pandas as pd

from utils_package.util_funcs import store_json, load_json
from utils_package.logger import get_logger

logger = get_logger()

def count_sources(data: pd.DataFrame):
  result = []
  
  for idx, row in data.iterrows():
    sources_str = row["sources"]
    if has_empty_sources(sources_str):
      result.append(None)
      continue
    sources_list = sources_str.split(",")
    sources_count = len(sources_list)
    result.append(sources_count)
    
  return result


def calculate_avg_nr_of_sources(filtered_counts: List[int]):
  avg = sum(filtered_counts) / len(filtered_counts)
  return avg


def filter_counts(source_counts: List[int]):
  return [elem for elem in source_counts if elem]
  
  
def has_empty_sources(sources_str):
  if not isinstance(sources_str, str):
    return True
  return not sources_str or not sources_str.strip()
    
    
  
if __name__ == "__main__":
  
  data_path = "data/pubhealth/train.tsv"
  
  source_counts_file = "data/pubhealth/stats/source_counts.json"
  
  data = pd.read_csv(data_path, sep="\t")
  
  if not os.path.isfile(source_counts_file):
    source_counts = count_sources(data)
    filtered_counts = filter_counts(source_counts)
    store_json(filtered_counts, source_counts_file, sort_keys=True)
    logger.info(f"Stored source counts data in '{source_counts_file}'")
  else:
    filtered_counts = load_json(source_counts_file)
    logger.info(f"Sentence source counts already exists in '{source_counts_file}'")

  avg_nr_of_sources = calculate_avg_nr_of_sources(filtered_counts)
  print(f"Average numbers of sources for the pubhealth dataset: {avg_nr_of_sources}")

