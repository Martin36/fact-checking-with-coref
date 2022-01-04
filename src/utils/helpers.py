

def encode_fever_text(input: str):
  output = input.replace('( ', '-LRB-')
  output = output.replace(' )', '-RRB-')
  output = output.replace(' - ', '-')
  output = output.replace(' :', '-COLON-')
  output = output.replace(' ,', ',')
  output = output.replace(" 's", "'s")
  output = output.replace(' ', '_')
  return output

def decode_fever_text(input: str):
  output = input.replace('-LRB-', '( ')
  output = output.replace('-RRB-', ' )')
  output = output.replace('-', ' - ')
  output = output.replace('-COLON-', ' :')
  output = output.replace(',', ' ,')
  output = output.replace("'s", " 's")
  output = output.replace('_', ' ')
  return output
