import pandas as pd
import random

# Load dataset
df = pd.read_csv("product_data.csv")
df.columns = df.columns.str.strip()  # Clean column names by stripping spaces

# Ensure 'Category' and 'Title' columns are present
if 'Category' not in df.columns or 'Title' not in df.columns:
    raise ValueError("'Category' or 'Title' column is missing!")

# Creating Positive and Negative Examples
df["Positive_Title"] = df.groupby("Category")["Title"].shift(-1)
df["Positive_Title"] = df["Positive_Title"].fillna(df.groupby("Category")["Title"].transform("first"))

# Shuffle Negative examples
df["Negative_Title"] = df["Title"].sample(frac=1, random_state=42).reset_index(drop=True)

# Keep necessary columns including 'Category'
df = df[["Category", "Title", "Positive_Title", "Negative_Title"]]  # Keep 'Category'

# Save modified dataset
df.to_csv("fine_tune_data.csv", index=False, encoding="utf-8")

print("Dataset prepared for BERT fine-tuning!")
