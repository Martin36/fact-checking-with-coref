import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

def create_numerical_subset(df: pd.DataFrame):
  filter = df.apply(filter_numerals, axis=1)
  return df[filter]

def filter_numerals(row):
  return contains_numerals(row["sent_en"])

def contains_numerals(text: str):
  doc = nlp(text)
  for token in doc:
    if token.pos_ == "NUM":
      return True
  return False


if __name__ == "__main__":
  data_file = "data/riksdagen/partiledardebatt-22-01-12.csv"
  df = pd.read_csv(data_file)
  df_numerical = create_numerical_subset(df)
  print(df_numerical.head())
  out_file = "data/riksdagen/partiledardebatt-numerical-22-01-12.csv"
  df_numerical.to_csv(out_file)