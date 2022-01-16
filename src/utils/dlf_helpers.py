from ast import literal_eval
import pandas as pd

def get_predicted_sentences(df: pd.DataFrame):
  """Gets the doc title, sentence id pairs from the document level fever predictions.

  Args:
      df (pd.DataFrame): The data frame part that contains sentence related a claim 

  Returns:
      List[tuple]: List of (document title, sentence id) pairs
  """
  
  pred_sents = []
  for _, row in df.iterrows():
    if row["y"] == 1:
      sent_tuple = literal_eval(row["page_sentence"])
      pred_sents.append(sent_tuple)
  return pred_sents
