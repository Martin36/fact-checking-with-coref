import torch, utils_package

from transformers import DebertaTokenizer, DebertaForSequenceClassification

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from utils_package.util_funcs import store_json, load_jsonl
from src.claim_verification.claim_verifier import ClaimVerifier

from src.data.fever_dataset import FEVERDataset
from src.utils.helpers import calc_accuracy, create_dirs_if_not_exist, tensor_dict_to_device
from src.utils.constants import FEVER_ID_2_LABEL, FEVER_LABEL_2_ID

logger = utils_package.logger.get_logger()

def compute_metrics(p):
  pred, labels = p
  # pred = np.argmax(pred, axis=1)

  accuracy = accuracy_score(y_true=labels, y_pred=pred)
  micro_recall = recall_score(y_true=labels, y_pred=pred, average="micro")
  micro_precision = precision_score(y_true=labels, y_pred=pred, average="micro")
  micro_f1 = f1_score(y_true=labels, y_pred=pred, average="micro")
  macro_recall = recall_score(y_true=labels, y_pred=pred, average="macro")
  macro_precision = precision_score(y_true=labels, y_pred=pred, average="macro")
  macro_f1 = f1_score(y_true=labels, y_pred=pred, average="macro")

  return {
    "accuracy": accuracy, 
    "micro_precision": micro_precision, 
    "micro_recall": micro_recall, 
    "micro_f1": micro_f1,
    "macro_recall": macro_recall, 
    "macro_precision": macro_precision, 
    "macro_f1": macro_f1
  }


def extract_title_id_pairs_from_evidence(evidence):
  return [[doc_title, sent_id] for _, _, doc_title, sent_id in evidence]


# TODO: Test this
def create_predicitions_data(pred_labels, labelled_data, prediction_data):
  
  for d_labelled, d_prediction, label_id in zip(labelled_data, prediction_data, pred_labels):
    d_labelled["predicted_label"] = FEVER_ID_2_LABEL[label_id]
    d_labelled["predicted_evidence"] = extract_title_id_pairs_from_evidence(d_prediction["evidence"])

  return labelled_data

    

def predict(claim_verifier, dataset, metrics_save_file=None, 
            predictions_save_file=None):
  batch_size = 8
  dataloader = DataLoader(dataset, batch_size=batch_size)
    
  pred_labels = []
  gold_labels = []
  for inputs in tqdm(dataloader):
    logits = claim_verifier.predict(inputs)
    pred_labels += claim_verifier.convert_logits_to_labels(logits)
    gold_labels += torch.squeeze(inputs["labels"]).tolist()
    
  metrics = compute_metrics((pred_labels, gold_labels))
  print("============ Evaluation Metrics ==============")
  for metric in metrics:
    print(f"{metric}: {metrics[metric]}")
  print("==============================================")
  print()  
  
  accuracy = calc_accuracy(pred_labels, gold_labels)
  logger.info(f"Accuracy for model '{model_name}' on dev set is: {accuracy}")

  labels = list(FEVER_LABEL_2_ID.keys())
  cls_report = classification_report(gold_labels, pred_labels, target_names=labels)
  print(cls_report)
  
  if metrics_save_file:
    create_dirs_if_not_exist(metrics_save_file)
    store_json(metrics, metrics_save_file, indent=2)
    print(f"Saved results in '{metrics_save_file}'")
    
  if predictions_save_file:
    # TODO
    predictions_data = create_predicitions_data(pred_labels, )
    create_dirs_if_not_exist(metrics_save_file)
    store_json(metrics, metrics_save_file, indent=2)
    print(f"Saved results in '{metrics_save_file}'")
    
    
if __name__ == "__main__":
  
  dev_data_path = "data/fever/dev_with_dlf.jsonl"
  labelled_data_path = "data/fever/dev.jsonl"
  train_data_path = "data/fever/train_with_evidence.jsonl"
  # model_name = "microsoft/deberta-v2-xlarge"
  # model_name = "models/document-level-FEVER/RTE-debertav2-MNLI"
  # model_name = "microsoft/deberta-v2-xlarge-mnli"
  # model_name = "microsoft/deberta-base-mnli"
  model_name = "models/deberta_base_mnli_finetuned/checkpoint-6000/"
  # model_out_path = "models/deberta_base_mnli_finetuned/"
  tokenizer_name = "microsoft/deberta-base-mnli"

  
  model = DebertaForSequenceClassification.from_pretrained(model_name, local_files_only=True)
  # tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
  tokenizer = DebertaTokenizer.from_pretrained(tokenizer_name)
 
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dev_dataset = FEVERDataset(dev_data_path, tokenizer=tokenizer)
  # train_dataset = FEVERDataset(train_data_path, tokenizer=tokenizer)

  claim_verifier = ClaimVerifier(model, device, 
                                 save_model_after_steps=1)
  
  save_file = "data/fever/claim_verification/deberta_base_mnli_finetined-ckpt_6000-dlf_predictions.json"
  labelled_data = load_jsonl(labelled_data_path)
  predict(claim_verifier, dev_dataset, save_file)
  
  # claim_verifier.train(train_dataset, dev_dataset)
  
  # claim_verifier.save_model(model_out_path, 1)
  
  