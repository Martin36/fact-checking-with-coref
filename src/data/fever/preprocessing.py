from utils_package.util_funcs import load_jsonl, store_jsonl

from src.utils.helpers import get_random_from_list


def create_limited_dev_set(dev_data, limit, out_file):
  limited_dataset = get_random_from_list(dev_data, n=limit)
  store_jsonl(limited_dataset, out_file)
  print(f"Stored mini dev set in '{out_file}'")
  
  
if __name__ == "__main__":
  dev_data = load_jsonl("data/fever/dev_with_evidence.jsonl")
  out_file = "data/fever/dev_with_evidence_500.jsonl"
  create_limited_dev_set(dev_data, 500, out_file)