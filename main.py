import streamlit as st
from vector_search_service import VectorSearchService
from search_interface import render_search_tab
from collection_manager import render_collection_manager_tab
from ebay_scraper_interface import render_ebay_scraper_tab


def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Semantic Product Search", layout="wide")
    
    st.title("SavantSearch")
    st.write("Search for products using natural language queries powered by BERT embeddings")
    
    # Initialize the service (with session state to avoid reloading the model)
    if 'search_service' not in st.session_state:
        with st.spinner("Loading model... This might take a minute."):
            st.session_state.search_service = VectorSearchService()
    
    service = st.session_state.search_service
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Search Products", "Manage Collections", "eBay Scraper"])
    
    with tab1:
        render_search_tab(service)
    
    with tab2:
        render_collection_manager_tab(service)
    
    with tab3:
        render_ebay_scraper_tab(service)
    
    # Footer
    st.divider()
    st.caption("Semantic Product Search Tool | Built with Streamlit, Transformers, and Qdrant")


if __name__ == "__main__":
    main()