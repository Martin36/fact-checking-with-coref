import pprint
# from util_funcs import load_jsonl
from utils_package import util_funcs

pp = pprint.PrettyPrinter(indent=2)

dev_path = "data/fever/dev.jsonl"
dev_data = util_funcs.load_jsonl(dev_path)

wiki_page_1_path = "data/fever/wiki-pages/wiki-001.jsonl"
wiki_page_1_data = util_funcs.load_jsonl(wiki_page_1_path)


pp.pprint(wiki_page_1_data[:5])