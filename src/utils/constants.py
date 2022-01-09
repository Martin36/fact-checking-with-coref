
# FEVER to 'deberta-v2-xlarge-mnli' mapping
# TODO: Rename this
label2id = {
  "REFUTES": 0,
  "NOT ENOUGH INFO": 1,
  "SUPPORTS": 2
}

# For 'deberta-v2-xlarge-mnli'
# TODO: Rename this
id2label = {
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
