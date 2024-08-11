import os
from dotenv import load_dotenv
import sys
from typing import List

from utils.files_handler import FileHandler
from pymilvus import MilvusClient, MilvusException, DataType
from sqlalchemy import create_engine
from langchain_community.vectorstores.singlestoredb import SingleStoreDB

from sentence_transformers import SentenceTransformer

class MilvusConnector:

    def __init__(self, local=True):
        """
        Instantiate the MilvusConnector class. 
        Choose between connecting to local or remote hosted instance of Milvus.
        Methods use standard MilvusClient methods and attributes, with some modication and abstraction.
        """
        file_handler = FileHandler()
        file_handler.get_config('vector_db_config.yaml')
        self.params_config = file_handler.config['MILVUS']
        self.db_name = self.params_config['DB_NAME']

        if local:
            try:
                self.client = MilvusClient(self.db_name)
                self_hosted_conn = self.client.is_self_hosted
                if self_hosted_conn:
                    print("Connection established to Milvus client!")
            except MilvusException:
                raise

        
        
        self.index_building_params = self.params_config['INDEX_BUILDING_PARAMS']
        self.search_params = self.params_config['SEARCH_PARAMS']
        
        file_handler.get_config('embeddings_config.yaml')
        self.embeddings_config = file_handler.config
        self.model_provider = self.embeddings_config['MODEL_PROVIDER']
        self.embedding_model_name = self.embeddings_config[self.model_provider]['EMBEDDING_MODEL']
        self.embeddings_dim = self.embeddings_config[self.model_provider]['MODEL_PARAMS']['dimension']

        # store schema if created
        self.collection_schema = None

    def list_collections(self):
        return self.client.list_collections()

    def check_collection(self, collection_name):
        """Check of collection exists"""
        return self.client.has_collection(collection_name=collection_name)
    
    def drop_collection(self, collection_name):
        return self.client.drop_collection(collection_name=collection_name)
    
    def create_doc_url_schema(self):
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False
        )

        # Add the fields to the schema for storing embeddings, id and url.
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embeddings", datatype=DataType.FLOAT_VECTOR, dim=self.embeddings_dim)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1024, is_primary=False)
        schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length =128, is_primary=False)
        schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=64, is_primary=False)

        self.collection_schema = schema

    def create_collection(self, collection_name):
        """Create collection based on simple logic."""
        # Drop collection if it already exists.
        if self.client.has_collection(collection_name=collection_name):
            print(f"{collection_name} already exists. Dropping.")
            self.client.drop_collection(collection_name=collection_name)
            print(f"{collection_name} dropped. Creating new collection.")

        # Create collection.
        if self.collection_schema is not None:
            print("Building Index")
            # Prepare Index parameters
            index_params = self.client.prepare_index_params()
            # add indexes
            index_params.add_index(
                field_name="embeddings",
                metric_type="COSINE",
                params={"nlist": 128}
            )
            
            # create collection with schema and index
            self.client.create_collection(
                collection_name=collection_name,
                schema=self.collection_schema,
                index_params=index_params
            )

        else:
            self.client.create_collection(
                collection_name=collection_name,
                dimension=self.embeddings_dim
            )
        
        if self.client.has_collection(collection_name=collection_name):
            return f"Collection {collection_name} created!"
        else:
            return f"Unable to create collection {collection_name}"
        
    def describe_collection(self, collection_name):
        return self.client.describe_collection(collection_name)
        
    def insert(self, collection_name, data):
        """Insert data to collection based on simple logic."""
        
        result = self.client.insert(
            collection_name=collection_name,
            data=data
        )

        print(result)
        
        return result
    
    def search(self, collection_name, query: str, fields: list, anns_field='vector'):
        """Conduct search on the knowledge base using vector similarity search."""
        model = SentenceTransformer(self.embedding_model_name)
        query_vector = model.encode([query])

        result = self.client.search(
            collection_name=collection_name,
            data=query_vector,
            anns_field=anns_field,
            limit=4,
            output_fields=fields
        )

        output_dict = {}
        for hits in result:
            for field in fields:
                temp_list = []
                for hit in hits:
                    # get the value of the output field specified in the search request.                
                    temp_list.append(hit['entity'][field])
                    output_dict[field] = temp_list

        return output_dict


class SingleStoreConnector:
        """
        Single Store db connector class.
        """
        def __init__(self):
            # AUTHENTICATION
            # store connection keys from .env file by loading it as environment variables.
            load_dotenv()
            # we will use the serverless connection method which only requires the uri and token (API key).
            self.single_store_url = os.environ['SINGLE_STORE_URL']
            self.engine = create_engine(self.single_store_url)
            print(f"""Connected to {self.engine.url}""")

            self.docsearch = None

        def create_table_from_documents(self,
                                        document: List, 
                                        embedding_model,
                                        table_name
                                        ):
            """
            A function to search collection using query vectors.
            :param db_name: The name of the database to use.
            :param document: the document to be encoded using the embedding model. Should be passed as a list.
            :param embedding_model: The embedding model to be used. Currently supports HunggingFace embedding models.
            :param table_name: The name of the table to be created.
            :param embedding_dim: The dimension of the embeddings to be used.
            """
            
            # Load documents to the store.
            self.docsearch = SingleStoreDB.from_documents(
                documents=document,
                embedding=embedding_model,
                table_name=table_name
            )

        def run_query(
                self, 
                query: str
                ):

            docs = self.docsearch.similarity_search(query)
            return docs    
        