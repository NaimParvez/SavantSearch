import requests
from bs4 import BeautifulSoup
import numpy as np
import pickle
import os
import time
import random
from transformers import BertTokenizer, AutoModel
import torch
import pandas as pd

class EbayScraper:
    def __init__(self, model_path="fine_tuned_bert_triplet.pt"):
        # Define headers with dynamic user-agent rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        ]
        self.headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        # Create a session for persistent connections
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Initialize tokenizer and model for embeddings
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
    
    def search_ebay(self, query, max_pages=3):
        """
        Search eBay for products based on the query
        """
        products = []

        for page in range(1, max_pages + 1):
            # Construct the URL
            url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}&_pgn={page}"
            print(f"Fetching page {page}: {url}")

            try:
                # Fetch the page content
                response = self.session.get(url)
                response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find all product listings
                listings = soup.select('.s-item__info')
                if not listings:
                    print(f"No listings found on page {page}")
                    break

                for listing in listings:
                    # Skip promotional or irrelevant listings
                    if "Shop on eBay" in listing.text:
                        continue

                    # Extract title, price, and link
                    title_elem = listing.select_one('.s-item__title')
                    price_elem = listing.select_one('.s-item__price')
                    link_elem = listing.select_one('a.s-item__link')

                    if title_elem and price_elem and link_elem:
                        title = title_elem.text.strip()
                        price = price_elem.text.strip()
                        link = link_elem.get('href')

                        # Ensure the link is valid
                        if not link.startswith("http"):
                            link = "https://www.ebay.com" + link

                        products.append({
                            'title': title,
                            'price': price,
                            'link': link,
                            'search_query': query
                        })

                # Add a random delay between requests
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                print(f"Error on page {page}: {e}")
                break

        return products
    
    def get_embedding(self, text):
        """Generate embedding for a text query"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()
    
    def scrape_and_create_vectors(self, search_queries, output_dir="./ebay_data"):
        """
        Scrape eBay for multiple queries and create vectors
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        all_products = []
        
        # Scrape products for each query
        for query in search_queries:
            print(f"Scraping eBay for '{query}'...")
            products = self.search_ebay(query)
            all_products.extend(products)
            print(f"Found {len(products)} products for '{query}'")
        
        # Create a DataFrame and remove duplicates
        df = pd.DataFrame(all_products)
        df.drop_duplicates(subset=['title'], inplace=True)
        print(f"Total unique products: {len(df)}")
        
        # Generate embeddings for each product
        product_titles = df['title'].tolist()
        product_embeddings = []
        
        for title in product_titles:
            embedding = self.get_embedding(title)
            product_embeddings.append(embedding)
        
        # Convert to numpy array
        product_embeddings = np.array(product_embeddings)
        
        # Save embeddings and titles
        embeddings_path = os.path.join(output_dir, "ebay_embeddings.npy")
        titles_path = os.path.join(output_dir, "ebay_titles.pkl")
        metadata_path = os.path.join(output_dir, "ebay_products.csv")
        
        np.save(embeddings_path, product_embeddings)
        with open(titles_path, 'wb') as f:
            pickle.dump(product_titles, f)
        
        # Save full metadata as CSV
        df.to_csv(metadata_path, index=False)
        
        print(f"Saved embeddings to {embeddings_path}")
        print(f"Saved titles to {titles_path}")
        print(f"Saved full product data to {metadata_path}")
        
        return {
            'embeddings_path': embeddings_path,
            'titles_path': titles_path,
            'metadata_path': metadata_path,
            'product_count': len(product_titles)
        }
    
    def get_sample_queries(self):
        """Return sample queries for eBay scraping"""
        return [
            "gaming laptop",
            "wireless headphones",
            "smartphone",
            "fitness tracker",
            "bluetooth speaker",
            "smart watch",
            "digital camera",
            "tablet"
        ]

# For testing the scraper directly
if __name__ == "__main__":
    scraper = EbayScraper()
    search_queries = scraper.get_sample_queries()
    result = scraper.scrape_and_create_vectors(search_queries)
    print(f"Created embeddings for {result['product_count']} products")