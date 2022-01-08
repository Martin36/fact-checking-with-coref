import torch

from transformers import BigBirdForTokenClassification, BigBirdModel, BigBirdTokenizer, BigBirdConfig

from src.lib.document_level_fever.sentence_selection_model import SentenceSelectionModel
from src.lib.document_level_fever.sentence_selection_dataset import FEVERDataset


class SentenceRetriever():
  
  def __init__(self) -> None:
    pass
  
  
  
if __name__ == "__main__":
  
  model_name = "sentence-selection-bigbird-base"
  device = "cuda" if torch.cuda.is_available() else "cpu"

  config = BigBirdConfig.from_pretrained(model_name)
  config.gradient_checkpointing = True
  config.num_labels = 1

  model = SentenceSelectionModel(model_name, config, device).to(device)

  tokenizer = BigBirdTokenizer.from_pretrained(model_name)
  
  eval_dataset = FEVERDataset(args.eval_file, tokenizer, mode="validation")
  eval_dataloader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, collate_fn = eval_dataset.collate_fn, batch_size=args.batch_size)
  evaluate(model, eval_dataloader, tokenizer, args.predict_filename)

