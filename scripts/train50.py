from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from evaluate import load
import numpy as np

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):#need to double check
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=preds, references=labels)


# Load pre-split data
train_df = pd.read_csv("../data//imdb_test_train_datasets/train/train_5050.csv")  # or train_5050.csv etc.
test_df = pd.read_csv("../data/imdb_test_train_datasets/test/test_5050.csv")

# Encode string labels into integers
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["label"])
test_df["label"] = label_encoder.transform(test_df["label"])

# Extract lists
train_texts = train_df["review"].tolist()
train_labels = train_df["label"].tolist()
val_texts = test_df["review"].tolist()
val_labels = test_df["label"].tolist()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
val_dataset = IMDBDataset(val_texts, val_labels, tokenizer)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
)

accuracy = load("accuracy")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./models/fine_tuned_bert_imdb/imdb_bert_50")
tokenizer.save_pretrained("./models/fine_tuned_bert_imdb/imdb_bert_50")