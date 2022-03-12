import requests, json, os
import pandas as pd
from tqdm import tqdm

API_KEY = os.environ.get("CLAIM_BUSTER_API_KEY")
api_base_url = "https://idir.uta.edu/claimbuster/api/v2/"
single_endpoint = "score/text/"
multiple_endpoint = "score/text/sentences/"

request_headers = {"x-api-key": API_KEY}


def create_data_with_predictions(df: pd.DataFrame, out_file: str):
  predictions = []

  for _, row in tqdm(df.iterrows(), total=len(df)):
    text = row["sent_en"]
    api_endpoint = api_base_url + single_endpoint + text
    response = requests.get(url=api_endpoint, headers=request_headers)
    res_body = response.json()
    if len(res_body["results"]) > 1:
      print(res_body)
    predictions.append(res_body["results"][0]["score"])

  df_predictions = pd.DataFrame(predictions, columns=['prediction'])
  
  df = pd.concat([df, df_predictions], axis=1)
  print(df.head())
  df.to_csv(out_file)


def create_sample_data_with_predictions(df: pd.DataFrame, out_file: str, 
                                        sample_size=50):
  df_sample = df.sample(sample_size)
  create_data_with_predictions(df_sample, out_file)
  

if __name__ == "__main__":
  data_file = "data/riksdagen/partiledardebatt-numerical-22-01-12-labeled.csv"
  out_file = "data/riksdagen/partiledardebatt-numerical-22-01-12-labeled-with-predictions.csv"
  df = pd.read_csv(data_file)
  create_data_with_predictions(df, out_file)
