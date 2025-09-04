import streamlit as st


def render_search_tab(service):
    """Render the search products tab"""
    st.header("Search Products")
    
    # Get available collections
    collections = service.list_collections()
    
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