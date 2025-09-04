import streamlit as st
import torch
from transformers import AutoModel, BertTokenizer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import pickle
import os
import pandas as pd
from eBay_Scraper import EbayScraper
from bs4 import BeautifulSoup
import time
import random


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

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

# Streamlit application
def main():
    st.set_page_config(page_title="Semantic Product Search", layout="wide")
    
    st.title("SavantSearch")
    st.write("Search for products using natural language queries powered by BERT embeddings")
    
    # Initialize the service (with session state to avoid reloading the model)
    if 'search_service' not in st.session_state:
        with st.spinner("Loading model... This might take a minute."):
            st.session_state.search_service = VectorSearchService()
    
    service = st.session_state.search_service
    
    # Get available collections
    collections = service.list_collections()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Search Products", "Manage Collections", "eBay Scraper"])
    
    with tab1:
        st.header("Search Products")
        
        # Select collection to search
        collection_name = "products"
        if collections:
            collection_name = st.selectbox(
                "Select collection to search:",
                options=collections,
                index=0 if "products" in collections else 0
            )
        
        collection_exists = service.collection_exists(collection_name)
        
        if not collection_exists:
            st.warning(f"⚠️ Collection '{collection_name}' not found. Please set up a collection in the 'Manage Collections' tab first.")
        else:
            # Search interface
            query = st.text_input("Enter your search query:", key="search_query")
            num_results = st.slider("Number of results to show:", min_value=1, max_value=50, value=10)
            
            if st.button("Search", key="search_button") and query:
                with st.spinner("Searching..."):
                    results = service.search_products(query, collection_name=collection_name, limit=num_results)
                
                if results:
                    st.subheader("Search Results")
                    for i, hit in enumerate(results, 1):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{i}. {hit.payload['title']}**")
                            
                            # Display additional metadata if available
                            if "price" in hit.payload:
                                st.write(f"Price: {hit.payload['price']}")
                            
                            if "link" in hit.payload:
                                st.markdown(f"[View on eBay]({hit.payload['link']})")
                        
                        with col2:
                            st.write(f"Score: {hit.score:.4f}")
                            st.progress(float(hit.score))
                        
                        st.divider()
                else:
                    st.info("No results found. Try a different query.")
    
    with tab2:
        st.header("Manage Vector Collections")
        
        # Show available collections
        st.subheader("Available Collections")
        if collections:
            for i, collection in enumerate(collections, 1):
                st.write(f"{i}. {collection}")
        else:
            st.info("No collections found.")
        
        st.divider()
        
        # Collection setup interface
        st.subheader("Set Up New Collection")
        
        # Collection name input
        new_collection_name = st.text_input("Collection Name:", value="products")
        
        # File upload for embeddings and titles
        st.write("Upload your vector data:")
        embeddings_file = st.file_uploader("Upload embeddings (NPY file)", type=["npy"])
        titles_file = st.file_uploader("Upload titles (PKL file)", type=["pkl"])
        metadata_file = st.file_uploader("Upload metadata (CSV file, optional)", type=["csv"])
        
        if st.button("Set Up Collection", type="primary"):
            if not embeddings_file or not titles_file:
                st.error("Please upload both embeddings and titles files.")
            else:
                # Save uploaded files temporarily
                with open("temp_embeddings.npy", "wb") as f:
                    f.write(embeddings_file.getbuffer())
                
                with open("temp_titles.pkl", "wb") as f:
                    f.write(titles_file.getbuffer())
                
                metadata_path = None
                if metadata_file:
                    metadata_path = "temp_metadata.csv"
                    with open(metadata_path, "wb") as f:
                        f.write(metadata_file.getbuffer())
                
                # Set up collection
                with st.spinner("Setting up collection..."):
                    result = service.setup_collection(
                        embeddings_path="temp_embeddings.npy", 
                        titles_path="temp_titles.pkl",
                        collection_name=new_collection_name,
                        metadata_path=metadata_path
                    )
                
                # Clean up temporary files
                os.remove("temp_embeddings.npy")
                os.remove("temp_titles.pkl")
                if metadata_path:
                    os.remove(metadata_path)
                
                st.success(result)
                st.rerun()  # Refresh the app to update collection list
                
        # For demo purposes, add option to use default files if available
        st.divider()
        st.subheader("Use Default Files")
        if os.path.exists("product_embeddings.npy") and os.path.exists("product_titles.pkl"):
            if st.button("Use Default Product Files"):
                with st.spinner("Setting up collection with default files..."):
                    result = service.setup_collection(collection_name="products")
                st.success(result)
                st.rerun()
        
        if os.path.exists("./ebay_data/ebay_embeddings.npy") and os.path.exists("./ebay_data/ebay_titles.pkl"):
            if st.button("Use eBay Product Files"):
                with st.spinner("Setting up collection with eBay files..."):
                    result = service.setup_collection(
                        embeddings_path="./ebay_data/ebay_embeddings.npy",
                        titles_path="./ebay_data/ebay_titles.pkl",
                        collection_name="ebay_products",
                        metadata_path="./ebay_data/ebay_products.csv"
                    )
                st.success(result)
                st.rerun()
    
    with tab3:
        st.header("eBay Product Scraper")
        st.write("Scrape product data from eBay, generate embeddings, and create searchable collections")
        
        # Initialize eBay scraper if not already in session state
        if 'ebay_scraper' not in st.session_state:
            with st.spinner("Initializing eBay scraper..."):
                st.session_state.ebay_scraper = EbayScraper()
        
        scraper = st.session_state.ebay_scraper
        
        # Get default sample queries
        sample_queries = scraper.get_sample_queries()
        
        # Input for search queries
        st.subheader("eBay Search Queries")
        st.write("Enter product categories to scrape from eBay (one per line):")
        
        default_queries = "\n".join(sample_queries[:3])  # Show just a few default queries
        search_queries = st.text_area("Search Queries:", value=default_queries, height=150)
        
        # Parse queries
        query_list = [q.strip() for q in search_queries.split("\n") if q.strip()]
        
        # Scraper options
        st.subheader("Scraper Options")
        col1, col2 = st.columns(2)
        with col1:
            max_pages = st.number_input("Max pages per query:", min_value=1, max_value=10, value=2)
        with col2:
            collection_name = st.text_input("Collection name:", value="ebay_products")
        
        # Start scraping button
        if st.button("Start Scraping", type="primary"):
            if not query_list:
                st.error("Please enter at least one search query.")
            else:
                # Override the search_ebay method to show progress
                original_search_method = scraper.search_ebay
                
                # Create a wrapper to track progress
                def search_with_progress(query, max_pages=max_pages):
                    progress_text = f"Scraping eBay for '{query}'..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    products = []

                    # --- SELENIUM SETUP ---
                    # This sets up a Chrome browser instance that Selenium will control
                    options = webdriver.ChromeOptions()
                    options.add_argument('--headless')  # Run Chrome in the background without a UI
                    options.add_argument('--no-sandbox')
                    options.add_argument('--disable-dev-shm-usage')
                    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
                    
                    try:
                        # Use a context manager to ensure the browser closes properly
                        with webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options) as driver:
                            for page in range(1, max_pages + 1):
                                st.write(f"Processing page {page} for '{query}' using Selenium...")
                                url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}&_pgn={page}"
                                driver.get(url)

                                # --- WAIT FOR THE CONTENT TO LOAD ---
                                # This is the most important step. We wait up to 10 seconds for the
                                # product listings (li.s-item) to become visible on the page.
                                WebDriverWait(driver, 10).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, "li.s-item"))
                                )

                                # Now that the page is fully loaded, get the HTML and parse it
                                soup = BeautifulSoup(driver.page_source, 'html.parser')

                                listings = soup.select('li.s-item')
                                for listing in listings:
                                    title_elem = listing.select_one('.s-item__title')
                                    price_elem = listing.select_one('.s-item__price')
                                    link_elem = listing.select_one('a.s-item__link')

                                    if title_elem and price_elem and link_elem and "Shop on eBay" not in title_elem.text:
                                        products.append({
                                            'title': title_elem.text.strip(),
                                            'price': price_elem.text.strip(),
                                            'link': link_elem.get('href'),
                                            'search_query': query
                                        })
                                
                                my_bar.progress((page / max_pages), text=f"Page {page}/{max_pages} for '{query}'")

                    except Exception as e:
                        st.error(f"A Selenium error occurred for '{query}': {str(e)}")

                    my_bar.progress(1.0, text=f"Completed '{query}': Found {len(products)} products")
                    return products
                
                # Replace the method temporarily
                scraper.search_ebay = search_with_progress
                
                try:
                    # Create a placeholder for overall progress
                    st.subheader("Scraping Progress")
                    progress_container = st.empty()
                    progress_container.info(f"Starting scraping for {len(query_list)} queries...")
                    
                    # Scrape and create vectors
                    all_products = []
                    for i, query in enumerate(query_list):
                        progress_container.info(f"Processing query {i+1}/{len(query_list)}: '{query}'")
                        products = scraper.search_ebay(query)
                        all_products.extend(products)
                        
                    # Create dataframe and generate embeddings
                    if all_products:
                        progress_container.info("Generating embeddings...")
                        
                        # Create DataFrame and remove duplicates
                        df = pd.DataFrame(all_products)
                        df.drop_duplicates(subset=['title'], inplace=True)
                        
                        # Generate embeddings (simplified for display)
                        product_titles = df['title'].tolist()
                        
                        embedding_progress = st.progress(0, text="Generating embeddings...")
                        product_embeddings = []
                        
                        for i, title in enumerate(product_titles):
                            embedding = scraper.get_embedding(title)
                            product_embeddings.append(embedding)
                            embedding_progress.progress((i + 1) / len(product_titles), 
                                                      text=f"Generating embeddings: {i+1}/{len(product_titles)}")
                        
                        # Convert to numpy array
                        product_embeddings = np.array(product_embeddings)
                        
                        # Save files
                        os.makedirs("./ebay_data", exist_ok=True)
                        embeddings_path = "./ebay_data/ebay_embeddings.npy"
                        titles_path = "./ebay_data/ebay_titles.pkl"
                        metadata_path = "./ebay_data/ebay_products.csv"
                        
                        np.save(embeddings_path, product_embeddings)
                        with open(titles_path, 'wb') as f:
                            pickle.dump(product_titles, f)
                        df.to_csv(metadata_path, index=False)
                        
                        # Create collection
                        progress_container.info("Creating vector collection...")
                        result = service.setup_collection(
                            embeddings_path=embeddings_path,
                            titles_path=titles_path,
                            collection_name=collection_name,
                            metadata_path=metadata_path
                        )
                        
                        # Show success message
                        st.success(f"Successfully created collection '{collection_name}' with {len(product_titles)} products!")
                        st.balloons()
                        
                        # Show sample of the collected data
                        st.subheader("Sample of Collected Products")
                        st.dataframe(df.head(10))
                        
                        # Show instructions for searching
                        st.info("Go to the 'Search Products' tab and select your new collection to start searching!")
                    else:
                        st.error("No products found. Try different search queries or check your internet connection.")
                
                finally:
                    # Restore original method
                    scraper.search_ebay = original_search_method
    
    # Footer
    st.divider()
    st.caption("Semantic Product Search Tool | Built with Streamlit, Transformers, and Qdrant")

if __name__ == "__main__":
    main()