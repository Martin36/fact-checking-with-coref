import argparse
import torch, utils_package

from transformers import DebertaTokenizer, DebertaForSequenceClassification

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from utils_package.util_funcs import store_json, load_jsonl, store_jsonl

from lib.fever.scorer import fever_score

from src.claim_verification.claim_verifier import ClaimVerifier
from src.data.fever.dataset import FEVERDataset
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
  

def print_metrics(metrics):
  print("============ Evaluation Metrics ==============")
  for metric in metrics:
    print(f"{metric}: {metrics[metric]}")
  print("==============================================")
  print()


def extract_title_id_pairs_from_evidence(evidence):
  if len(evidence) == 0: return evidence
  return [[doc_title, sent_id] for _, _, doc_title, sent_id in evidence]


def create_predicitions_data(pred_labels, labelled_data, prediction_data):

  # labelled_data = labelled_data[:8]
  # prediction_data = prediction_data[:8]

  for d_labelled, d_prediction, label_id in zip(labelled_data, prediction_data, pred_labels):
    d_labelled["predicted_label"] = FEVER_ID_2_LABEL[label_id]
    first_evidence_set = d_prediction["evidence"][0]
    d_labelled["predicted_evidence"] = extract_title_id_pairs_from_evidence(first_evidence_set)

  return labelled_data


def create_test_data(pred_labels, prediction_data):

  # test_data = test_data[:8]
  # prediction_data = prediction_data[:8]
  result = []

  for d_prediction, label_id in zip(prediction_data, pred_labels):
    predicted_label = FEVER_ID_2_LABEL[label_id]
    first_evidence_set = d_prediction["evidence"][0]
    predicted_evidence = extract_title_id_pairs_from_evidence(first_evidence_set)
    res_obj = {
      "id": d_prediction["id"],
      "predicted_label": predicted_label,
      "predicted_evidence": predicted_evidence
    }
    result.append(res_obj)

  return result


def predict(claim_verifier, dataset, labelled_data, 
            metrics_save_file=None, predictions_save_file=None,
            incorrect_predictions_file=None):
  batch_size = 8
  dataloader = DataLoader(dataset, batch_size=batch_size)

  pred_labels = []
  gold_labels = []
  for inputs in tqdm(dataloader):
    logits = claim_verifier.predict(inputs)
    pred_labels += claim_verifier.convert_logits_to_labels(logits)
    gold_labels += torch.squeeze(inputs["labels"]).tolist()
    # break

  metrics = compute_metrics((pred_labels, gold_labels))
  print_metrics(metrics)

  show_cls_report(gold_labels, pred_labels)

  predictions = create_predicitions_data(
    pred_labels, labelled_data, dataset.data)
  
  metrics["fever"] = get_fever_metrics(predictions)

  if metrics_save_file:
    store_json(metrics, metrics_save_file, indent=2)
    print(f"Saved metrics in '{metrics_save_file}'")

  if predictions_save_file:
    store_jsonl(predictions, predictions_save_file)
    print(f"Saved predictions in '{predictions_save_file}'")
    
  if incorrect_predictions_file:
    store_incorrect_predictions(predictions, incorrect_predictions_file)
    

def store_incorrect_predictions(predictions, incorrect_predictions_file):
  incorrect_predictions = get_incorrect_predictions(predictions)
  store_jsonl(incorrect_predictions, incorrect_predictions_file)
  print(f"Saved incorrect predictions in '{incorrect_predictions_file}'")


def get_incorrect_predictions(predictions):
  incorrect_predictions = []
  for pred_obj in predictions:
    if pred_obj["predicted_label"] != pred_obj["label"]:
      incorrect_predictions.append(pred_obj)
  return incorrect_predictions  


def predict_test_data(claim_verifier, dataset,
            predictions_save_file=None):
  batch_size = 8
  dataloader = DataLoader(dataset, batch_size=batch_size)

  pred_labels = []
  for inputs in tqdm(dataloader):
    logits = claim_verifier.predict(inputs)
    pred_labels += claim_verifier.convert_logits_to_labels(logits)

  predictions = create_test_data(pred_labels, dataset.data)

  if predictions_save_file:
    store_jsonl(predictions, predictions_save_file)
    print(f"Saved predictions in '{predictions_save_file}'")
  

def show_cls_report(gold_labels, pred_labels):
  labels = list(FEVER_LABEL_2_ID.keys())
  cls_report = classification_report(gold_labels, pred_labels, target_names=labels)
  print(cls_report)
  

def get_fever_metrics(predictions):
  strict_score, label_accuracy, precision, recall, f1 = fever_score(predictions)
  print("============ FEVER Score =====================")
  print(f"Strict score: {strict_score}")
  print(f"Label accuracy: {label_accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1: {f1}")
  print("==============================================")
  print()

  fever_metrics = {
    "strict_score": strict_score,
    "label_accuracy": label_accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
  }

  return fever_metrics



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--sent_retr_data_path', type=str, help="Path to the file contaning the data from document level fever")
  parser.add_argument('--labelled_data_path', type=str, default=None, help="(Optional) Path to the labelled dataset file e.g. dev.jsonl")
  parser.add_argument('--output_folder', type=str, help="The folder to store the output data")
  args = parser.parse_args()

  # model_name = "microsoft/deberta-v2-xlarge"
  # model_name = "models/document-level-FEVER/RTE-debertav2-MNLI"
  # model_name = "microsoft/deberta-v2-xlarge-mnli"
  # model_name = "microsoft/deberta-base-mnli"
  model_name = "models/deberta_base_mnli_finetuned/checkpoint-6000/"
  # model_out_path = "models/deberta_base_mnli_finetuned/"
  tokenizer_name = "microsoft/deberta-base-mnli"

  model = DebertaForSequenceClassification.from_pretrained(model_name, local_files_only=True)

  tokenizer = DebertaTokenizer.from_pretrained(tokenizer_name)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dataset = FEVERDataset(args.sent_retr_data_path, tokenizer=tokenizer)

  claim_verifier = ClaimVerifier(model, device)

  if args.labelled_data_path:    
    metrics_save_file = args.output_folder + "deberta_base_mnli_finetined-ckpt_6000-dlf_metrics.json"
    predictions_save_file = args.output_folder + "deberta_base_mnli_finetined-ckpt_6000-dlf_predictions.jsonl"
    incorrect_predictions_file = args.output_folder + "deberta_base_mnli_finetined-ckpt_6000-dlf_incorrect_predictions.jsonl"
    labelled_data = load_jsonl(args.labelled_data_path)
    predict(claim_verifier, dataset, labelled_data, metrics_save_file, 
            predictions_save_file, incorrect_predictions_file)
  else:
    predictions_save_file = args.output_folder + "deberta_base_mnli_finetined-ckpt_6000-dlf_predictions.jsonl"
    predict_test_data(claim_verifier, dataset, predictions_save_file)
