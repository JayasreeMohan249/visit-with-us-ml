import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define constant for the dataset path
RAW_DATA_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
try:
    df = pd.read_csv(RAW_DATA_PATH)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: {RAW_DATA_PATH} not found. Please ensure the file is in the correct directory.")
    exit()

# Drop unnecessary columns
# CustomerID is an identifier, Designation might be too granular or not relevant for package purchase
df = df.drop(columns=['CustomerID'], errors='ignore')

# Drop 'Unnamed: 0' if it exists, an extraneous index column
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    print("Dropped 'Unnamed: 0' column.")

# Handle missing values by dropping rows with any missing data
initial_rows = df.shape[0]
df.dropna(inplace=True)
rows_after_dropping_na = df.shape[0]
if initial_rows - rows_after_dropping_na > 0:
    print(f"Dropped {initial_rows - rows_after_dropping_na} rows due to missing values.")


# Define target column
target_col = 'ProdTaken' # As per Data Description

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Stratify for classification tasks
)

# Save the splits locally as CSV files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation complete. Train and test splits saved as CSV files.")
print(f"Xtrain shape: {Xtrain.shape}, ytrain shape: {ytrain.shape}")
print(f"Xtest shape: {Xtest.shape}, ytest shape: {ytest.shape}")
