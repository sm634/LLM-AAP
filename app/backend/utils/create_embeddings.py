from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
from sentence_transformers import SentenceTransformer
from utils.files_handler import FileHandler
from typing import List, Union, Dict
import numpy as np


file_handler = FileHandler()
file_handler.get_config('embeddings_config.yaml')
embeddings_config = file_handler.config

def get_splitter():
    # set up text splitter
    if embeddings_config['CHUNKER'] == 'LANGCHAIN_SPLITTER':
        CHUNKER = embeddings_config['LANGCHAIN_SPLITTER']
        text_splitter = CHUNKER['splitter']
        chunking_params = CHUNKER['chunking_params']

        if text_splitter == 'RecursiveCharacterTextSplitter':
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunking_params['chunk_size'],
                chunk_overlap=chunking_params['chunk_overlap'],
                length_function=len
            )
        elif text_splitter == 'SpacyTextSplitter':
            splitter = SpacyTextSplitter(
                chunk_size=chunking_params['chunk_size'],
                chunk_overlap=chunking_params['chunk_overlap'],
                length_function=len
            )
    
    return splitter

def chunk_text(text):
    """
    A function that takes in text data and chunks it using the chunker to create documents.
    """
    text_splitter = get_splitter()
    documents = text_splitter.create_documents([text])
    return documents


def chunk_texts_list(
        input_data: Union[Dict, List],
        ):
    """
    Chunks up a list of text. With the linked_data option, it can also return the associated data, if provided from a dictionary.
    """

    document_content = []    
    if isinstance(input_data, list):

        for text in input_data:
            chunks = chunk_text(text)
            for chunk in chunks:
                document_content.append(chunk.page_content) 
            
    elif isinstance(input_data, dict):

        for k,v in list(input_data.items()):
            chunks = chunk_text(v)
            for chunk in chunks:
                document_content.append((k, chunk.page_content))
    
    return document_content


# embeding model 
def MiniLML6V2EmbeddingFunction(input_texts: List[str]):
    model_provider = embeddings_config['MODEL_PROVIDER']
    if model_provider == 'HUGGING_FACE':
        embedding_model_name = embeddings_config[model_provider]['EMBEDDING_MODEL']
        model = SentenceTransformer(embedding_model_name)
        chunks = [text for text in input_texts]
        return model.encode(chunks)
