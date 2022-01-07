import unittest

from src.utils.helpers import calc_accuracy

class TestHelpers(unittest.TestCase):
  
  def test_encode_fever_text(self):
    pass
  
  def test_decode_fever_text(self):
    pass
  
  def test_create_input_str(self):
    pass
  
  def test_tensor_dict_to_device(self):
    pass
  
  def test_calc_accuracy(self):
    pred_labels = [1, 2, 1, 0]
    gold_labels = [1, 2, 0, 1]
    correct = 0.5
    output = calc_accuracy(pred_labels, gold_labels)
    self.assertEqual(output, correct, "Should calculate the correct accuracy")
    
    
if __name__ == "__main__":
  unittest.main()
    