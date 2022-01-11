import torch

from transformers import BigBirdForTokenClassification, BigBirdModel, BigBirdTokenizer, BigBirdConfig

from src.lib.document_level_fever.sentence_selection_model import SentenceSelectionModel
from src.lib.document_level_fever.sentence_selection_dataset import FEVERDataset


class SentenceRetriever():
  
  def __init__(self, model) -> None:
    self.model = model
    pass
  
  def predict(self, batch):
    with torch.no_grad():
      self.model.eval()
      # predict labels for each token in a document
      output_i = model(**batch["model_input"]).logits.squeeze()
      output_i_list = output_i.cpu().tolist()
      predictions = output_i_list if type(output_i_list[0]) is list else [output_i_list]
      labels = batch["labels"].cpu().tolist()
      input_ids = batch["model_input"]["input_ids"].cpu().tolist()
      pages = batch["pages"]
      sent_ids = batch["sent_ids"]
      claim_ids = batch["claim_id"]

    pass
  
# TODO: 
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

