import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_brand_products(brand_url):
    """Scrapes product data from a brand's website."""
    try:
        response = requests.get(brand_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        products = []
        product_elements = soup.find_all('div', class_='product')  # Adjust selector
        for product_element in product_elements:
            title_element = product_element.find('h2', class_='title')  # Adjust selector
            price_element = product_element.find('span', class_='price')  # Adjust selector
            if title_element and price_element:
                title = title_element.text.strip()
                price = price_element.text.strip()
                products.append({'title': title, 'price': price})
        return pd.DataFrame(products)
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    brand_url = "https://motionview.com.bd/"
    product_data = scrape_brand_products(brand_url)
    if not product_data.empty:
        product_data.to_csv("data/scraped_products.csv", index=False)
        print("Product data scraped successfully and saved to scraped_products.csv")
    else:
        print("Failed to scrape product data.")