from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
df = pd.read_csv("/mnt/data/clean_imdb_dataset.csv")

# Define features and labels (adjust column names as needed)
X = df["text"]
y = df["label"]

# Step 1: First split off 20% for train + val
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.8, random_state=1)

# Step 2: Split the 20% into equal 10% train and 10% validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

# Output shapes
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
