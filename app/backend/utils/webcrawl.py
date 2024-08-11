"""
A simple script that will crawl a provided list of urls using Beautiful Soup
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor
import re


def get_urls_list(file_name='watsonx.gov.txt'):
    """
    Reads the file containing the list of urls and returns them as a list.
    """
    file_path = f'data/websites/{file_name}'
    with open(file=file_path, mode='r') as f:
        urls = f.read()
        urls_list = urls.split(',')
        urls_list = [url.replace('\n', '') for url in urls_list]
    
    return urls_list

def get_webpage_content(file_name='watsonx.gov.txt'):
    """
    A simple function that takes in a list of urls then returns a list of 
    texts extracted from those urls using BautifulSoup. 
    return: Dict, url and associated content from that page.
    """
    soup_url_texts = {}
    
    urls = get_urls_list(file_name=file_name)   

    for url in tqdm(urls):
        page = urlopen(url)
        html = page.read().decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")
        soup_text = soup.get_text()
        soup_text = soup_text.replace("\n", "")
        # Remove excess whitespaces and newlines
        cleaned_text = re.sub(r'\s+', ' ', soup_text).strip()        
        # Remove HTML tags
        cleaned_text = re.sub(r'<.*?>', '', cleaned_text)
        soup_url_texts[url] = soup_text
    
    return soup_url_texts


def extract_text_from_website(url):
    # Fetch the HTML content of the webpage
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all text from the webpage
        raw_text = soup.get_text()
        
        # Remove excess whitespaces and newlines
        cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()
        
        # Remove HTML tags
        cleaned_text = re.sub(r'<.*?>', '', cleaned_text)
        
        return cleaned_text
    else:
        # If request was unsuccessful, return None
        print("Failed to retrieve webpage:", response.status_code)
        return None

def scrape_multiple_websites(urls):
    with ThreadPoolExecutor() as executor:
        # Submit each URL for scraping concurrently
        futures = [executor.submit(extract_text_from_website, url) for url in urls]
        # Retrieve results as they become available
        results = [future.result() for future in futures]
        
    # Store text from each URL as a separate item in a list
    return results
