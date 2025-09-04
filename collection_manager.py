import streamlit as st
import os


def render_collection_manager_tab(service):
    """Render the manage collections tab"""
    st.header("Manage Vector Collections")
    
    # Get available collections
    collections = service.list_collections()
    
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