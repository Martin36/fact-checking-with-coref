import unittest

from src.utils.translator import Translator

class TestTranslator(unittest.TestCase):
  
  def test_get_filename(self):
    translator = Translator()
    input = "Partiledardebatt 12 januari 2022"
    expected = "partiledardebatt-12-januari-2022.json"
    output = translator.get_filename(input)
    self.assertEqual(output, expected, "Transforms date into filename correctly")


if __name__ == "__main__":
  unittest.main()