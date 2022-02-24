import unittest

from src.data.riksdagen.create_numerical_subset import contains_numerals


class TestCreateNumericalSubset(unittest.TestCase):
  
  def test_contains_numerals(self):
    input = "The number of patients in the country's hospitals has increased by almost 40 percent in the past week, and the risk is high for a very large number of sick leaves."
    correct = True
    output = contains_numerals(input)
    self.assertEqual(output, correct, "Should return True if text contains arabic numerals")
    
    input = "It just does not take another four years!"
    correct = True
    output = contains_numerals(input)
    self.assertEqual(output, correct, "Should return True if text contains worded numerals")

    input = "Some text without a numeral"
    correct = False
    output = contains_numerals(input)
    self.assertEqual(output, correct, "Should return False if text doesn't contain numerals")

if __name__ == "__main__":
  unittest.main()