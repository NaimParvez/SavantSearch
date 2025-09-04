import torch
from transformers import AutoModel, BertTokenizer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import pickle
import os
import pandas as pd


class VectorSearchService:
    def __init__(self, model_path="fine_tuned_bert_triplet.pt", qdrant_path="./qdrant_data"):
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Create a custom model class that matches the saved structure
        class CustomBertModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = AutoModel.from_pretrained("bert-base-uncased")
            
            def forward(self, **inputs):
                return self.bert(**inputs)
        
        # Initialize the custom model and load state dict
        self.model = CustomBertModel()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Connect to Qdrant
        self.client = QdrantClient(path=qdrant_path)
        
    def get_embedding(self, text):
        """Generate embedding for a text query"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()
    
    def setup_collection(self, embeddings_path="product_embeddings.npy", titles_path="product_titles.pkl", collection_name="products", metadata_path=None):
        """Initialize the Qdrant collection with product embeddings"""
        # Load precomputed embeddings and metadata
        embeddings = np.load(embeddings_path)
        with open(titles_path, 'rb') as f:
            titles = pickle.load(f)
        
        # Load additional metadata if available
        metadata_df = None
        if metadata_path and os.path.exists(metadata_path):
            metadata_df = pd.read_csv(metadata_path)
        
        # Create a new collection for products
        vector_size = embeddings.shape[1]  # Dimension of your BERT embeddings
        
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        
        # Prepare points to upload
        points = []
        for i, (vector, title) in enumerate(zip(embeddings, titles)):
            payload = {"title": title}
            
            # Add additional metadata if available
            if metadata_df is not None:
                if i < len(metadata_df):
                    row = metadata_df.iloc[i]
                    # Add price and link if available
                    if 'price' in row:
                        payload["price"] = row['price']
                    if 'link' in row:
                        payload["link"] = row['link']
            
            points.append(models.PointStruct(
                id=i,
                vector=vector.tolist(),
                payload=payload
            ))
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection_name,
                points=points[i:i+batch_size]
            )
            
        return f"Uploaded {len(points)} vectors to Qdrant collection '{collection_name}'"
    
    def search_products(self, query_text, collection_name="products", limit=10):
        """Search for products similar to the query text"""
        query_vector = self.get_embedding(query_text)
        
        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=limit
        )
        
        return [hit for hit in search_results]

    def collection_exists(self, collection_name="products"):
        """Check if the specified collection exists"""
        try:
            collections = self.client.get_collections()
            for collection in collections.collections:
                if collection.name == collection_name:
                    return True
            return False
        except Exception:
            return False

    def list_collections(self):
        """List all available collections"""
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception:
            return []