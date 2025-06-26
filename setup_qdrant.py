import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pickle

# Load your precomputed embeddings and metadata
embeddings = np.load('product_embeddings.npy')
with open('product_titles.pkl', 'rb') as f:
    titles = pickle.load(f)

# Initialize Qdrant client - local mode
client = QdrantClient(path="./qdrant_data")  # Store data locally

# Create a new collection for your products
collection_name = "products"
vector_size = embeddings.shape[1]  # Dimension of your BERT embeddings

client.recreate_collection(
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
    client.upsert(
        collection_name=collection_name,
        points=points[i:i+batch_size]
    )

print(f"Uploaded {len(points)} vectors to Qdrant collection '{collection_name}'")