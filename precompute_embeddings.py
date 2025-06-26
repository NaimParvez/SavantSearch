import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
import pickle

print("Starting embedding pre-computation...")

# Define the model
class BertTripletModel(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BertTripletModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

# Set device
device = torch.device("cpu")  # Change to "cuda" if you want to use GPU

# Load model and tokenizer
print("Loading model and tokenizer...")
model = BertTripletModel()
checkpoint = torch.load('fine_tuned_bert_triplet.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('fine_tuned_bert_tokenizer')

# Function to encode titles
def encode_title(title, model, tokenizer):
    inputs = tokenizer(
        title,
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        embedding = model(input_ids, attention_mask)
    
    return embedding

# Load data
print("Loading product data...")
df = pd.read_csv("fine_tune_data.csv")
all_products = df['Title'].tolist()

# Save product list for reference
with open('product_titles.pkl', 'wb') as f:
    pickle.dump(all_products, f)

# Pre-compute embeddings
print(f"Pre-computing embeddings for {len(all_products)} products...")
all_embeddings = []

for title in tqdm(all_products, desc="Computing embeddings"):
    emb = encode_title(title, model, tokenizer)
    # Convert to numpy for easier serialization
    all_embeddings.append(emb.cpu().numpy())

# Save embeddings
print("Saving embeddings...")
with open('product_embeddings.pkl', 'wb') as f:
    pickle.dump(all_embeddings, f)

# Optional: Also save as numpy array
embeddings_array = np.array([e.squeeze(0) for e in all_embeddings])
np.save('product_embeddings.npy', embeddings_array)

print("Pre-computation complete!")
print(f"Saved embeddings for {len(all_products)} products")
print("Files created: 'product_titles.pkl', 'product_embeddings.pkl', 'product_embeddings.npy'")