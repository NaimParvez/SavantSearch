import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the dataset
df = pd.read_csv("fine_tune_data.csv")

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom Dataset Class for triplet learning
class ProductSearchDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=64):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Get the anchor, positive, and negative titles
        anchor_title = str(row['Title'])
        positive_title = str(row['Positive_Title'])
        negative_title = str(row['Negative_Title'])
        
        # Encode separately to maintain proper structure
        anchor_encoding = self.tokenizer(
            anchor_title,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        positive_encoding = self.tokenizer(
            positive_title,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        negative_encoding = self.tokenizer(
            negative_title,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
            'negative_input_ids': negative_encoding['input_ids'].squeeze(0),
            'negative_attention_mask': negative_encoding['attention_mask'].squeeze(0),
            'category': row['Category']  # Keep category for reference
        }

# Define a triplet loss model using BERT embeddings
class BertTripletModel(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BertTripletModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token embedding as the sentence representation
        return outputs.last_hidden_state[:, 0, :]
    
    def get_embedding(self, input_ids, attention_mask):
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)

# Triplet loss function
def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = F.pairwise_distance(anchor, positive, p=2)
    distance_negative = F.pairwise_distance(anchor, negative, p=2)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()

# Prepare dataset
dataset = ProductSearchDataset(df, tokenizer)

# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = ProductSearchDataset(train_df, tokenizer)
val_dataset = ProductSearchDataset(val_df, tokenizer)

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize the model
model = BertTripletModel()
optimizer = AdamW(model.parameters(), lr=2e-5)

# Add a learning rate scheduler
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
def train_model(model, train_loader, optimizer, scheduler, epoch=1):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        # Move tensors to device
        anchor_input_ids = batch['anchor_input_ids'].to(device)
        anchor_attention_mask = batch['anchor_attention_mask'].to(device)
        positive_input_ids = batch['positive_input_ids'].to(device)
        positive_attention_mask = batch['positive_attention_mask'].to(device)
        negative_input_ids = batch['negative_input_ids'].to(device)
        negative_attention_mask = batch['negative_attention_mask'].to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass for all three inputs
        anchor_embeddings = model(anchor_input_ids, anchor_attention_mask)
        positive_embeddings = model(positive_input_ids, positive_attention_mask)
        negative_embeddings = model(negative_input_ids, negative_attention_mask)
        
        # Calculate triplet loss
        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
    
    return total_loss / len(train_loader)

# Validation function
def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move tensors to device
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)
            
            # Forward pass
            anchor_embeddings = model(anchor_input_ids, anchor_attention_mask)
            positive_embeddings = model(positive_input_ids, positive_attention_mask)
            negative_embeddings = model(negative_input_ids, negative_attention_mask)
            
            # Calculate loss
            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# Run training for a specified number of epochs
epochs = 3
best_val_loss = float('inf')

for epoch in range(1, epochs + 1):
    # Train the model
    train_loss = train_model(model, train_loader, optimizer, scheduler, epoch)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
    
    # Evaluate the model
    val_loss = evaluate_model(model, val_loader)
    print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
        }, 'fine_tuned_bert_triplet.pt')
        
        # Save the tokenizer for future use
        tokenizer.save_pretrained('fine_tuned_bert_tokenizer')
        print(f"Model saved at epoch {epoch} with validation loss: {val_loss:.4f}")

print("Training complete!")

# Function to encode titles into embeddings
def encode_title(title, model, tokenizer):
    model.eval()
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
    
    return embedding.cpu().numpy()

# Example of using the model for inference
def find_similar_products(query_title, product_titles, model, tokenizer, top_k=5):
    # Encode the query title
    query_embedding = encode_title(query_title, model, tokenizer)
    
    # Encode all product titles
    product_embeddings = []
    for title in product_titles:
        embedding = encode_title(title, model, tokenizer)
        product_embeddings.append(embedding)
    
    # Calculate similarities (cosine similarity)
    similarities = []
    for i, embedding in enumerate(product_embeddings):
        similarity = F.cosine_similarity(
            torch.tensor(query_embedding),
            torch.tensor(embedding),
            dim=1
        ).item()
        similarities.append((i, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k similar products
    top_similar = [(product_titles[idx], sim) for idx, sim in similarities[:top_k]]
    return top_similar

# Use this at the end if you want to test the model
# Example:
# test_query = "School Backpack for Children"
# test_products = df['Title'].tolist()
# similar_products = find_similar_products(test_query, test_products, model, tokenizer)
# print("Similar products to:", test_query)
# for product, similarity in similar_products:
#     print(f"- {product} (Similarity: {similarity:.4f})")