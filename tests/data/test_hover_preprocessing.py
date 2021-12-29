import unittest

from src.data.hover_preprocessing import escape_single_quotes, remove_hyperlinks

class TestHoverPreprocessing(unittest.TestCase):
  
  def test_remove_hyperlinks(self):
    input = 'Canal Street Confidential is the eighth <a href="studio%20album">studio album</a> by American rapper <a href="Currensy">Currensy</a>.'
    correct = 'Canal Street Confidential is the eighth studio album by American rapper Currensy.'
    output = remove_hyperlinks(input)
    self.assertEqual(output, correct, "Should remove hyperlinks correctly")
    
  def test_escape_single_quotes(self):
    input = 'On August 28, 2015 the album\'s first single "Lil Wayne was released.'
    correct = 'On August 28, 2015 the album\'\'s first single "Lil Wayne was released.'
    output = escape_single_quotes(input)
    self.assertEqual(output, correct, "Should escape single quotes correctly")

if __name__ == "__main__":
  unittest.main()