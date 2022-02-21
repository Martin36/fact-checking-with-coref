import requests, json, os
import pandas as pd
from tqdm import tqdm

API_KEY = os.environ.get("CLAIM_BUSTER_API_KEY")
api_base_url = "https://idir.uta.edu/claimbuster/api/v2/"
single_endpoint = "score/text/"
multiple_endpoint = "score/text/sentences/"

request_headers = {"x-api-key": API_KEY}


data_file = "data/riksdagen/partiledardebatt-22-01-12.csv"
df = pd.read_csv(data_file)

results = []
input = ""
nr_of_samples = 50
df_sample = df.sample(nr_of_samples)

for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
  text = row["sent_en"]
  api_endpoint = api_base_url + single_endpoint + text
  response = requests.get(url=api_endpoint, headers=request_headers)
  res_body = response.json()
  if len(res_body["results"]) > 1:
    print(res_body)
  results.append({
    "text_en": text,
    "text_sv": row["sent_sv"],
    "speaker": row["speaker"],
    "score": res_body["results"][0]["score"]
  })

out_file = "data/riksdagen/partiledardebatt-labeled-sample-22-01-12.csv"
df_results = pd.DataFrame(results)
df_results.to_csv(out_file)
