import pandas as pd

def filter_statements(data: pd.DataFrame):
  
  print(data.shape)
  rows_to_keep = data["sent_sv"] != "Fru talman!"
  data = data[rows_to_keep]
  print(data.shape)
  rows_to_keep = data["sent_sv"] != "(Applåder)"
  data = data[rows_to_keep]  
  print(data.shape)
  return data
      
      
if __name__ == "__main__":
  data_file = "data/riksdagen/partiledardebatt-22-01-12-cleaned.csv"
  out_file = "data/riksdagen/partiledardebatt-22-01-12-cleaned-filtered.csv"
  data = pd.read_csv(data_file)
  data = filter_statements(data)
  data.to_csv(out_file)