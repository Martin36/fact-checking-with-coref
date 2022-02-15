import six
from google.cloud import translate_v2 as translate
from tqdm import tqdm
from utils_package.util_funcs import load_jsonl, store_json
from utils_package.logger import get_logger

logger = get_logger()

class Translator():
  
  def __init__(self, target_lang="en") -> None:
    self.target_lang = target_lang
    self.translate_client = translate.Client()
  
  
  def translate_debate_dataset(self, dataset):
    for d in dataset:
      d = self.translate_debate(d)
      self.store_debate(d)
    return dataset
  
  def translate_debate(self, debate_data):
    for statement in tqdm(debate_data["statements"]):
      paragraphs = statement["text"].split("\n")
      translated_paragraphs = []
      for p in paragraphs:
        translated_p = self.translate_text(p)
        translated_paragraphs.append(translated_p)
      statement["text_en"] = "\n".join(translated_paragraphs)    
    return debate_data
  
  
  def store_debate(self, d):
    file_name = self.get_filename(d["date"])
    folder_path = "data/riksdagen/"
    out_file = folder_path + file_name
    store_json(d, out_file)
    logger.info(f"Stored data in '{out_file}'")
    
    
  def translate_text(self, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://cloud.google.com/translate/docs/languages
    """

    if isinstance(text, six.binary_type):
      text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = self.translate_client.translate(text, target_language=self.target_lang, source_language="")

    return result["translatedText"]

    # print(u"Text: {}".format(result["input"]))
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))


  def get_filename(self, date):
    date = date.lower()
    return "-".join(date.split()) + ".json"
  

  def list_languages(self):
    """Lists all available languages."""

    results = self.translate_client.get_languages()

    for language in results:
      print(u"{name} ({language})".format(**language))


if __name__ == "__main__":

  data_path = "data/riksdagen/partiledardebatter.jsonl"
  data = load_jsonl(data_path)
  
  translator = Translator()
  data_with_translation = translator.translate_debate_dataset(data)
  
  # out_path = "data/riksdagen/partiledardebatt-22-01-12_en.json"
  # store_json(data_with_translation, out_path)
  # logger.info(f"Stored data in '{out_path}'")
