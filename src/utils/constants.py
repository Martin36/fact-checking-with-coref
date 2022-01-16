
# FEVER to 'deberta-v2-xlarge-mnli' mapping
FEVER_LABEL_2_ID = {
  "REFUTES": 0,
  "NOT ENOUGH INFO": 1,
  "SUPPORTS": 2
}

FEVER_ID_2_LABEL = {
  0: "REFUTES",
  1: "NOT ENOUGH INFO",
  2: "SUPPORTS"
}


# For 'deberta-v2-xlarge-mnli'
DEBERTA_MNLI_LABEL_2_ID = {
  0: "CONTRADICTION",
  1: "NEUTRAL",
  2: "ENTAILMENT"
}

# For FEVER labels
# TODO: Rename this
LABEL_2_IDX = {
  "SUPPORTS": 0, 
  "REFUTES": 1, 
  "NOT ENOUGH INFO": 2
}

# For FEVER labels
# TODO: Rename this
IDX_2_LABEL = {
  0: "SUPPORTS", 
  1: "REFUTES", 
  2: "NOT ENOUGH INFO"
}
