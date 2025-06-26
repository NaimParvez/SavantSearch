import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from qdrant_client import QdrantClient, models
import numpy as np

def update_model_and_qdrant(scraped_data_path, model_path, tokenizer_path, qdrant_client):
    """Updates the BERT model and Qdrant database."""
    df = pd.read_csv(scraped_data_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertModel.from_pretrained(model_path)
    embeddings = []
    for title in df['title']:
        inputs = tokenizer(title, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        embeddings.append(embedding)
    points = [
        models.PointStruct(
            id=i, 
            vector=embedding.tolist()[0], 
            payload={"title": df['title'][i]}
        )
        for i, embedding in enumerate(embeddings)
    ]
    qdrant_client.upsert(collection_name="products", points=points)
    print("Model and Qdrant database updated successfully.")

if __name__ == "__main__":
    scraped_data_path = "data/scraped_products.csv"
    model_path = "fine_tuned_bert_triplet.pt"
    tokenizer_path = "fine_tuned_bert_tokenizer"
    client = QdrantClient(":memory:")
    update_model_and_qdrant(scraped_data_path, model_path, tokenizer_path, client)