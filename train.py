import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
    Trainer,
    TrainingArguments,
)

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
                batch["text"], padding=True, truncation=True, max_length=self.max_length
            )

        self.data.rename_column_("intent", "labels")
        self.data.map(tokenize, batched=True, batch_size=None)
        self.data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        return self.data


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg):

    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Load Dataset
    clinc_data = clinc_dataset(
        max_length=cfg.processing.max_length, batch_size=cfg.processing.batch_size
    )

    print(clinc_data.data)

if __name__ == "__main__":
    main()
