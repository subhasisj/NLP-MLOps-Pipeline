import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from sklearn import metrics
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TextClassificationPipeline, Trainer,
                          TrainingArguments)

RANDOM_SEED = 34
wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class clinc_dataset:
    def __init__(self, max_length=None, batch_size=None) -> None:
        self.data = load_dataset("clinc_oos", "plus")
        self.intents = self.data["train"].features["intent"]
        self.num_labels = self.data["train"].features["intent"].num_classes
        self.max_length = max_length
        self.batch_size = batch_size

    def tokenize_dataset(self, tokenizer):
        def tokenize(batch):
            return tokenizer(
                batch["text"], padding=True, truncation=True,# max_length=self.max_length
            )

        self.data.rename_column_("intent", "labels")
        self.data = self.data.map(tokenize, batched=True, batch_size=None)#self.batch_size)
        self.data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        return self.data


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels, pred)})

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="weighted")
    precision = precision_score(y_true=labels, y_pred=pred, average="weighted")
    f1 = f1_score(y_true=labels, y_pred=pred, average="weighted")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg):

    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Load Dataset
    clinc_data = clinc_dataset(
        max_length=cfg.processing.max_length, batch_size=cfg.processing.batch_size
    )

    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.tokenizer,
        num_labels=clinc_data.data["train"].features["intent"].num_classes,
    ).to(device)

    # Tokenize the data
    clinc_encoded = clinc_data.tokenize_dataset(tokenizer=tokenizer)
    logging_steps = (
        len(clinc_encoded["train"]) // cfg.training.per_device_train_batch_size
    )

    # Load Training Arguments
    args = TrainingArguments(
        report_to="wandb",
        output_dir=f"results-{cfg.model.name}",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        warmup_steps=250,
        # label_smoothing_factor = 0.8,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        num_train_epochs=cfg.training.max_epochs,
        # save_strategy='epoch',
        save_steps=logging_steps * 2,
        save_total_limit=cfg.training.save_total_limit,
        seed=RANDOM_SEED,
        learning_rate=cfg.training.learning_rate,
        fp16=cfg.training.fp16,
        weight_decay=0.01,
        metric_for_best_model="f1",
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        logging_steps=logging_steps,
        run_name=cfg.training.run_name,
    )

    # Start Training
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=clinc_encoded["train"],
        eval_dataset=clinc_encoded["validation"],
        compute_metrics=compute_metrics,
    )

    # Train pre-trained model
    trainer.train()

    # Evaluate
    trainer.evaluate()

    test_preds = trainer.predict(clinc_encoded["test"])
    labels = clinc_encoded["test"].features["labels"].names

    y_preds = np.argmax(test_preds.predictions, axis=1)
    y_test = np.array(clinc_encoded["test"]["labels"])

    print(classification_report(y_test, y_preds, target_names=labels))
    y_test = np.array(clinc_encoded["test"]["labels"])

    # Log CM to Wandb
    wandb.log({"test_set_cm": wandb.sklearn.plot_confusion_matrix(y_test, y_preds)})
    wandb.finish()

    # Save model and Tokenizer
    MODEL_DIR = os.path.join("models", f"{cfg.model.name}")

    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    main()
