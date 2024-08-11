from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from connectors.elasticsearch_connector import ElasticSearchConnector
from utils.files_handler import FileHandler
import json
import re


class URLParser:

    def __init__(self) -> None:
        self.file_handler = FileHandler()

        # datasource handling from websites.
        urls_file = self.file_handler.get_json_input_file('source_urls.json')
        self.urls = urls_file['urls']
        self.parser = 'html.parser'

    def extract_text_from_website(self, url):
        # Fetch the HTML content of the webpage
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, self.parser)
            
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
        
    def scrape_datasource_urls(self):
        with ThreadPoolExecutor() as executor:
            # Submit each URL for scraping concurrently.
            futures = [executor.submit(self.extract_text_from_website, url) for url in self.urls]
            # Retrieve results as the become available.
            results = [future.result() for future in futures]
        
        # Store text from each URL as a separte item in a list
        return results


class ElasticDataPrep(URLParser):

    def __init__(self) -> None:

        # Initialize globally as it will be reused.
        self.search = ElasticSearchConnector()

        # crawler
        """To be written once the crawler and preprocessing is required."""
        
        # store documents
        self.documents = None
        self.chunked_documents = None
        self.documents_embeddings = None


    def load_documents(self, document_path="data/input/FoundationModels-Data-b.csv"):
        """
        A function that loads the required document using langchain loader.
        """
        loader=CSVLoader(document_path)
        models_info_data= loader.load()
        
        self.documents = models_info_data

        print("Documents Loaded")


    def chunk_documents(self):
        """
        A function used to chuck documents
        """
        if self.documents is None:
            self.load_documents()
            documents = self.documents
        else:
            documents = self.documents

        text_splitter = CharacterTextSplitter(separator="\ufeff")
        # Split text into chunks
        chunked_documents = text_splitter.split_documents(documents)
        self.chunked_documents = chunked_documents

        print("Chunking performed on Documents")

    @staticmethod
    def get_model_info_dict(chunked_content):
        """
        A helper function to extract the models info into a dictionary
        """
        # Split the string into lines
        lines = chunked_content.split('\n')

        # Create a dictionary from the lines
        model_info = {}
        for line in lines:
            # Split on the first colon found
            key_value = line.split(':', 1)
            if len(key_value) == 2:
                key = key_value[0].strip()
                value = key_value[1].strip()
                if key.lower() == "size (parameters)" or key.lower() == "context length":
                    value = value.replace(" ", "")
                    value = int(value)
                elif key.lower() == "price":
                    value = float(value)
                model_info[key] = value

        # Output the dictionary
        return model_info


    def create_docs_embeddings(self):
        """
        A function that takes a list of langchain.core.documents.base.Document type and creates embeddings for each, along with id.
        
        :param: chunked_docs: list of langchain.core.documents.base.Document type.
        :return: List[embeddings]
        """
        if self.chunked_documents is None:
            self.chunk_documents()
            chunked_docs = self.chunked_documents
        else:
            chunked_docs = self.chunked_documents

        embeddings = []
        for doc in chunked_docs:
            document = doc.page_content
            emb = self.search.get_embedding(document)
            embeddings.append(emb)
        
        self.documents_embeddings = embeddings

        print("Documents Embeddings Created")


    def docs_to_json(self, output_file_name):
        """
        A function that converts the relevant chunked documents with 
        langchain.core.documents.base.Document content, embeddings and id into a json format that is suitable
        to upload to a elasticsearch index.
        """

        # output
        output = []

        # Set-up the relevant processes to retrieve the write documents and embeddings to store as json.
        if self.chunked_documents is None:
            self.chunk_documents()
            chunked_docs = self.chunked_documents
        else:
            chunked_docs = self.chunked_documents

        for doc in chunked_docs:

            # To be used as a template and overwritten + filled for each model.
            document_json_structure = {
                'model_id': None,
                'model_info': None,
                'document': None
            }

            chunked_content = doc.page_content
            model_info_dict = self.get_model_info_dict(chunked_content)
            
            document_json_structure['model_id'] = model_info_dict['ID']
            document_json_structure['model_info'] = model_info_dict
            document_json_structure['document'] = chunked_content

            output.append(document_json_structure)

        output_file_path = f'data/input/elasticsearch/{output_file_name}.json'

        with open(output_file_path, 'w') as f:
            json.dump(output, f, indent=4)

        print(f"{output_file_name} saved as Json")

        return output


    def run_data_prep_pipeline(self, index_name='models_infos'):
        # prepare the documents
        documents = self.docs_to_json(output_file_name=index_name)
        # upload the documents with embeddings to the index
        self.search.create_index(index_name=index_name)
        self.search.insert_documents(
            documents=documents,
            passage_for_embedding='document',
            index_name=index_name
        )
