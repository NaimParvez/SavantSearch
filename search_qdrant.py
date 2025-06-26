import torch
from transformers import AutoModel, BertTokenizer, AutoModelForSequenceClassification
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import pickle

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
    
    def setup_collection(self, embeddings_path="product_embeddings.npy", titles_path="product_titles.pkl"):
        """Initialize the Qdrant collection with product embeddings"""
        # Load precomputed embeddings and metadata
        embeddings = np.load(embeddings_path)
        with open(titles_path, 'rb') as f:
            titles = pickle.load(f)
            
        # Create a new collection for products
        collection_name = "products"
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
            points.append(models.PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={"title": title}
            ))
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection_name,
                points=points[i:i+batch_size]
            )
            
        print(f"Uploaded {len(points)} vectors to Qdrant collection '{collection_name}'")
    
    def search_products(self, query_text, limit=10):
        """Search for products similar to the query text"""
        query_vector = self.get_embedding(query_text)
        
        search_results = self.client.search(
            collection_name="products",
            query_vector=query_vector.tolist(),
            limit=limit
        )
        
        return [(hit.payload["title"], hit.score) for hit in search_results]

# Simple command-line interface
def main():
    # Initialize the service
    service = VectorSearchService()
    
    # First-time setup (run this only once)
    setup_needed = input("Do you need to set up the Qdrant collection? (y/n): ").lower() == 'y'
    if setup_needed:
        service.setup_collection()
    
    # Interactive search loop
    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        results = service.search_products(query)
        
        print("\nSearch Results:")
        for i, (title, score) in enumerate(results, 1):
            print(f"{i}. {title} (Score: {score:.4f})")

if __name__ == "__main__":
    main()