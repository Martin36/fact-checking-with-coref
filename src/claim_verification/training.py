import numpy as np
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def compute_metrics(p):
  pred, labels = p
  pred = np.argmax(pred, axis=1)

  accuracy = accuracy_score(y_true=labels, y_pred=pred)
  recall = recall_score(y_true=labels, y_pred=pred, average="micro")
  precision = precision_score(y_true=labels, y_pred=pred, average="micro")
  f1 = f1_score(y_true=labels, y_pred=pred, average="micro")

  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train_with_trainer(model, train_dataset, dev_dataset):
  out_dir = "models/deberta_base_mnli_finetuned"
  training_args = TrainingArguments(
    out_dir, 
    overwrite_output_dir=True, 
    evaluation_strategy="steps", 
    eval_steps=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-5,
    weight_decay=0,
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=10,
    gradient_checkpointing=True
  )
  
  trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
  )
  
  trainer.train()
