from flask import Flask, render_template, request
from transformers import BertTokenizer, BertModel
from qdrant_client import QdrantClient, models
import torch
import numpy as np
import pandas as pd
from scripts.scraper import scrape_brand_products
from scripts.update_model import update_model_and_qdrant
from scripts.analytics import SearchAnalytics
from scripts.assistant import ShoppingAssistant

app = Flask(__name__)
tokenizer = BertTokenizer.from_pretrained("fine_tuned_bert_tokenizer")
model = BertModel.from_pretrained("fine_tuned_bert_triplet.pt")
client = QdrantClient(":memory:")  # Replace with your actual Qdrant client connection string
analytics = SearchAnalytics()
assistant = ShoppingAssistant()

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    assistant_response = None
    if request.method == "POST":
        brands = request.form.getlist("brands")
        query = request.form["query"]
        category = request.form["category"]
        collection = request.form["collection"]
        assistant_input = request.form.get("assistant_input")

        # Scrape and update model
        if brands:
            scraped_data = pd.DataFrame()
            for brand_url in brands:
                brand_data = scrape_brand_products(brand_url)
                scraped_data = pd.concat([scraped_data, brand_data], ignore_index=True)

            if not scraped_data.empty:
                scraped_data.to_csv("data/scraped_products.csv", index=False)
                update_model_and_qdrant("data/scraped_products.csv", "fine_tuned_bert_triplet.pt", "fine_tuned_bert_tokenizer", client)

        # Perform search
        query_embedding = get_query_embedding(query)
        search_results = client.search(collection_name="products", query_vector=query_embedding[0].tolist(), limit=10)

        # Apply filters
        filtered_results = []
        for hit in search_results:
            product = hit.payload
            if (not category or product.get("category") == category) and (not collection or product.get("collection") == collection):
                filtered_results.append(product)

        results = filtered_results

        # Log search
        analytics.log_search(query)

        # Get assistant response
        if assistant_input:
            assistant_response = assistant.get_response(assistant_input)

    # Analytics data
    top_searches = analytics.get_top_searches()
    click_through_rate = analytics.get_click_through_rate()
    search_counts = analytics.get_search_counts()

    return render_template("index.html", results=results, top_searches=top_searches, click_through_rate=click_through_rate, search_counts=search_counts, assistant_response=assistant_response)

def get_query_embedding(query):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

if __name__ == "__main__":
    app.run(debug=True)