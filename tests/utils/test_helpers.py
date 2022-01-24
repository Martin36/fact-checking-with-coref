import unittest

from src.utils.helpers import calc_accuracy, get_fever_doc_lines_test

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
    
  def test_get_fever_doc_lines(self):
    input = '0\tA College preparatory course is a means by which college bound high school students may better meet the more stringent scholastic requirements for entry into colleges and universities .\n1\tStudents taking college-preparatory courses may have an increased quantity of classwork , and expectations to achieve are at a higher level .\n2\tThe GPA -LRB- grade point average -RRB- weight for college-preparatory courses may have more value for college entry programs than regular courses .\n3\tCollege prep courses are particularly appropriate for providing the academic background needed to succeed in a degree program at a college or university .\n4\tAbove college-preparatory in difficulty is honors , where the advanced structure while similar in many ways to college prep , requires even more effort from the student .\thonors\tHonors course\n5\tIn many schools , a student can move from college-preparatory courses to Advanced Placement AP courses , if they attain a certain average .\tAdvanced Placement AP\n\n\tAdvanced Placement Program\n6\t'
    output = get_fever_doc_lines_test(input)
    
    
if __name__ == "__main__":
  unittest.main()
    