# SavantSearch 🔍

A sophisticated semantic product search engine that leverages fine-tuned BERT embeddings and vector similarity search to understand natural language queries and find relevant products.

## ✨ Features

- **Semantic Search**: Natural language product search using fine-tuned BERT embeddings
- **eBay Integration**: Real-time product scraping and search from eBay
- **Vector Database**: Fast similarity search using Qdrant vector database
- **Web Interface**: User-friendly Streamlit web application
- **Collection Management**: Create and manage multiple product collections
- **Model Training**: Custom BERT fine-tuning with triplet loss for better product matching

## 🏗️ Architecture

The system consists of several key components:

1. **Fine-tuned BERT Model**: Custom trained BERT model for product embeddings
2. **Qdrant Vector Database**: Stores and searches product embeddings efficiently
3. **eBay Scraper**: Extracts product information from eBay search results
4. **Streamlit Interface**: Interactive web application for search and management
5. **Collection Manager**: Organizes products into searchable collections

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/NaimParvez/SavantSearch.git
   cd SavantSearch
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv env
   # On Windows
   env\Scripts\activate
   # On macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the vector database**

   ```bash
   python setup_qdrant.py
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

The application will be available at `http://localhost:8501`

## 📖 Usage

### Search Products

1. Navigate to the "Search Products" tab
2. Enter your natural language query (e.g., "wireless bluetooth headphones for running")
3. View semantically relevant results with similarity scores

### Manage Collections

1. Go to the "Manage Collections" tab
2. Create new collections or switch between existing ones
3. Add products manually or import from CSV files

### eBay Scraper

1. Use the "eBay Scraper" tab to search and import products from eBay
2. Enter search terms and specify the number of pages to scrape
3. Products are automatically processed and added to your collection

## 🧠 Model Training

The system uses a fine-tuned BERT model trained with triplet loss for better product similarity understanding.

### Training Your Own Model

1. **Prepare training data**

   ```bash
   python prep_data.py
   ```

2. **Train the model**

   ```bash
   python train_bert.py
   ```

3. **Precompute embeddings**
   ```bash
   python precompute_embeddings.py
   ```

### Training Data Format

The training data should be in CSV format with columns:

- `Title`: Anchor product title
- `Positive_Title`: Similar product title
- `Negative_Title`: Dissimilar product title

## 🗂️ Project Structure

```
├── main.py                      # Main Streamlit application
├── vector_search_service.py     # Core search service
├── search_interface.py          # Search UI components
├── collection_manager.py        # Collection management interface
├── ebay_scraper_interface.py    # eBay scraper UI
├── eBay_Scraper.py             # eBay scraping functionality
├── train_bert.py               # Model training script
├── precompute_embeddings.py    # Embedding generation
├── setup_qdrant.py             # Vector database setup
├── prep_data.py                # Data preparation utilities
├── requirements.txt            # Python dependencies
├── data/                       # Training and product data
├── ebay_data/                  # eBay scraped data
├── qdrant_data/               # Vector database storage
├── scripts/                   # Additional utility scripts
└── templates/                 # HTML templates
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Model Configuration
MODEL_PATH=fine_tuned_bert_triplet.pt
TOKENIZER_PATH=fine_tuned_bert_tokenizer/

# eBay Scraper Configuration
MAX_SCRAPE_PAGES=5
SCRAPE_DELAY=2
```

### Model Files

- `fine_tuned_bert_triplet.pt`: Fine-tuned BERT model weights
- `fine_tuned_bert_tokenizer/`: Tokenizer configuration files
- `product_embeddings.npy`: Precomputed product embeddings
- `product_titles.pkl`: Product titles corresponding to embeddings

## 📊 Performance

- **Search Speed**: Sub-second similarity search across thousands of products
- **Accuracy**: High semantic understanding with fine-tuned embeddings
- **Scalability**: Efficient vector search with Qdrant database
- **Memory Usage**: Optimized for local deployment

## 🛠️ Development

### Adding New Features

1. **Custom Scrapers**: Extend the scraping functionality in `scripts/scraper.py`
2. **New Interfaces**: Add UI components in the respective interface files
3. **Model Improvements**: Modify training parameters in `train_bert.py`

### Testing

Run the test suite:

```bash
python -m pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Dependencies

### Core Libraries

- **Streamlit**: Web application framework
- **Transformers**: BERT model and tokenization
- **Qdrant Client**: Vector database interaction
- **PyTorch**: Deep learning framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

### Additional Libraries

- **BeautifulSoup4**: Web scraping
- **Requests**: HTTP requests
- **Scikit-learn**: Machine learning utilities
- **tqdm**: Progress bars

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**

   ```
   RuntimeError: Error loading model
   ```

   **Solution**: Ensure the model file `fine_tuned_bert_triplet.pt` exists and is not corrupted.

2. **Qdrant Connection Error**

   ```
   ConnectionError: Cannot connect to Qdrant
   ```

   **Solution**: Run `python setup_qdrant.py` to initialize the vector database.

3. **Memory Issues**
   ```
   OutOfMemoryError during embedding computation
   ```
   **Solution**: Reduce batch size in embedding computation or use a machine with more RAM.

### Performance Optimization

- Use GPU acceleration for faster embedding computation
- Adjust Qdrant configuration for better search performance
- Implement embedding caching for frequently searched queries

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Hugging Face for the pre-trained BERT models
- Qdrant team for the excellent vector database
- Streamlit team for the amazing web framework
- The open-source community for various tools and libraries

## 📞 Support

For support, please open an issue on GitHub or contact the maintainers.

---

**Built with ❤️ for semantic product search**
