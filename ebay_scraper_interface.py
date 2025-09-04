import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from eBay_Scraper import EbayScraper


def render_ebay_scraper_tab(service):
    """Render the eBay scraper tab"""
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
            # Create a wrapper to track progress
            def search_with_progress(query, max_pages=max_pages):
                progress_text = f"Scraping eBay for '{query}'..."
                my_bar = st.progress(0, text=progress_text)
                
                products = []

                # --- SELENIUM SETUP ---
                options = webdriver.ChromeOptions()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
                
                try:
                    with webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options) as driver:
                        for page in range(1, max_pages + 1):
                            st.write(f"Processing page {page} for '{query}' using Selenium...")
                            url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}&_pgn={page}"
                            driver.get(url)

                            WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "li.s-item"))
                            )

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
            
            try:
                # Create a placeholder for overall progress
                st.subheader("Scraping Progress")
                progress_container = st.empty()
                progress_container.info(f"Starting scraping for {len(query_list)} queries...")
                
                # Scrape and create vectors
                all_products = []
                for i, query in enumerate(query_list):
                    progress_container.info(f"Processing query {i+1}/{len(query_list)}: '{query}'")
                    products = search_with_progress(query)
                    all_products.extend(products)
                    
                # Create dataframe and generate embeddings
                if all_products:
                    progress_container.info("Generating embeddings...")
                    
                    # Create DataFrame and remove duplicates
                    df = pd.DataFrame(all_products)
                    df.drop_duplicates(subset=['title'], inplace=True)
                    
                    # Generate embeddings
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
            
            except Exception as e:
                st.error(f"An error occurred during scraping: {str(e)}")