from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import random

random.seed(42)
torch.manual_seed(42)

# Custom dataset for reviews
class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}
    
df = pd.read_csv("../data/directional_expectation_data.csv") #change to relevant filepath
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])


model = BertForSequenceClassification.from_pretrained("./models/imdb_bert_90") #change to relevant filepath
tokenizer = BertTokenizer.from_pretrained("./models/imdb_bert_90") #change to relevant filepath
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare dataset and dataloader
texts = df["review"].tolist()
dataset = ReviewDataset(texts, tokenizer)
loader = DataLoader(dataset, batch_size=16)  # adjust batch size as needed

# Inference
all_preds = []
all_confidences = []

with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confs = torch.max(probs, dim=1).values

        all_preds.extend(preds.cpu().tolist())
        all_confidences.extend(confs.cpu().tolist())

# Decode predictions
predicted_labels = label_encoder.inverse_transform(all_preds)

# Save results to DataFrame
df["predicted_sentiment"] = predicted_labels
df["confidence"] = all_confidences

# Export results
df.to_csv("../data/predictions/direxp_predictions.csv", index=False) #change to relevant filepath