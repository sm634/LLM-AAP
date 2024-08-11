# from ibm_watson import DiscoveryV2
# from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os
from utils.files_handler import FileHandler
import json
import time

class WatsonDiscoveryV2Connector:

    def __init__(self):
        """
        Watson Discovery V2 connector class.
        """
        # we will get the connection keys from the .env file by loading it to the environment variables.
        load_dotenv()
        self.authenticator = IAMAuthenticator(os.environ['WATSON_DISCOVERY_APIKEY'])
        self.discovery_instance = DiscoveryV2(
            version=os.environ['WATSON_DISCOVERY_VERSION'],
            authenticator=self.authenticator
        )
        self.discovery_instance.set_service_url(os.environ['WATSON_DISCOVERY_URL'])
        self.project_id = os.environ['WATSON_DISCOVERY_PROJECT_ID']

        # get passage details from config
        file_handler = FileHandler()
        file_handler.get_config(config_file_name='elasticsearch_config')
        self.config = file_handler.config

        # get parameter values after parsing config.
        self.passages_config = self.config['WATSON_DISCOVERY_V2']['passages']
        self.max_per_document = self.passages_config['max_per_document']
        self.characters = self.passages_config['characters']

        self.response = None  # to be reassigned with response json packet.

    def query_response(self, query, collection_ids: list):
        """Query results from Discovery collections.
        :param query: The query used for search.
        :param collection_ids: the set of collections to send the query request to."""

        response = self.discovery_instance.query(
            project_id=self.project_id,
            collection_ids=collection_ids,
            passages=self.passages_config,
            natural_language_query=query
        ).get_result()

        self.response = response

    def __get_results(self):
        """
        A hidden method that fetches the results of running the query.
        :return: the result json.
        """
        assert self.response is not None, "Please run the 'get_query_response' method before using this method."
        return self.response['results']

    def __get_kv_from_result(self, key: str):
        """
        Returns a list of all the items form a metadata. For instance the returned 'result' from the response,
        which is generated using the __get_results(self) method may or may not have certain key-value pairs.
        For instance, subtitle or table may appear as a key that can be accessed to retrieve the associated value for
        certain retrieved response results or not.
        :param key: Str the key to be searched for in the results output.
        :return: List of all values associated with the key (metadata)
        """
        # get all results
        results = self.__get_results()
        # store output.
        output = []

        for i in range(0, len(results)):
            try:
                output.append(results[i][key])
            except KeyError:
                # exception needed as not all desired keys will be retrieved for all retrieved data.
                output.append(f"THERE IS NO {key} FOR THIS DATA")

        return output

    def get_document_ids(self):
        """
        A function that grabs the document ids from the results of the query response.
        :return: List[document ids]
        """
        document_ids = self.__get_kv_from_result(key='document_id')
        return document_ids

    def get_result_metadata(self):
        """
        A function that grabs the result metadata from the results of the query response.
        :return: List[result metadata]
        """
        result_metadata = self.__get_kv_from_result(key='result_metadata')
        return result_metadata

    def get_subtitle(self):
        """
        A function that grabs the subtitles from the results of the query response.
        :return: List[subtitles]
        """
        subtitles = self.__get_kv_from_result(key='subtitle')
        return subtitles

    def get_document_passages(self):
        """
        A function that grabs the document passages from the results of the query response.
        :return: List[passages]
        """
        passages = self.__get_kv_from_result(key='document_passages')
        return passages

    def get_text(self):
        """
        A function that grabs the text from the results of the query response.
        :return: List[text]
        """
        text = self.__get_kv_from_result(key='text')
        return text

    def get_table(self):
        """
        A function that grabs the table data from the results of the query response.
        :return: List[table]
        """
        table = self.__get_kv_from_result(key='table')
        return table



class ElasticSearchConnector:
    def __init__(self, 
                 connection_type='basic_auth',
                 use_embedding_model=True,
                 ):
        """
        param: connection_type, str: values include [local, basic_auth, cloud_api_key, cert_api_key]"""
        if connection_type == 'local':
            self.es = Elasticsearch('http://localhost:9200')  # <-- connection options need to be added here
        elif connection_type == 'cloud_api_key':
            # get creds.
            load_dotenv()
            # connect
            self.es = Elasticsearch(
                cloud_id=os.environ['ELASTIC_CLOUD_ID'],
                api_key=os.environ['ELASTIC_API_KEY']
            )
        elif connection_type == 'cert_api_key':
            # get creds.
            load_dotenv()
            # connect
            self.es = Elasticsearch(
                os.environ['ELASTICSEARCH_URL'],
                ca_certs=os.environ['ELASTICSEARCH_CERT_PATH'],
                api_key=os.environ['ELASTICSEARCH_API_KEY']
            )
        elif connection_type == 'basic_auth':
            # get creds.
            load_dotenv()
            # connect
            self.es = Elasticsearch(
                os.environ['ELASTICSEARCH_URL'],
                ca_certs=os.environ['ELASTICSEARCH_CERT_PATH'],
                basic_auth=(
                    os.environ['ELASTICSEARCH_USERNAME'], 
                    os.environ['ELASTICSEARCH_PASSWORD']
                    )
                )
        
        if use_embedding_model:
            file_handler = FileHandler()
            file_handler.get_config(
                config_file_name='embeddings_config'
                )
            embedding_model_config = file_handler.config

            self.model_provider = embedding_model_config['MODEL_PROVIDER']
            self.model_name = embedding_model_config[self.model_provider]['EMBEDDING_MODEL']
            
            if self.model_provider == 'HUGGING_FACE':
                self.model = SentenceTransformer(self.model_name)
            elif self.model_provider == 'ELASTICSEARCH':
                self.model = self.model_name

        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        print(client_info.body)

    ## The following deploy_elser() method in search.py follows a few steps to download and install the ELSER v2 model, 
    ## and to create a pipeline that uses it to populate the elser_embedding field defined above.
    def deploy_elser(
            self, 
            fields_to_embed: list
            ):
        # download ELSER v2
        if self.model_provider == 'ELASTICSEARCH':
            self.es.ml.put_trained_model(model_id=self.model_name,
                                         input={'field_names': fields_to_embed}
                                        )
            
            # wait until ready
            while True:
                status = self.es.ml.get_trained_models(model_id=self.model_name,
                                                    include='definition_status')
                if status['trained_model_configs'][0]['fully_defined']:
                    # model is ready
                    break
                time.sleep(1)

            # deploy the model
            self.es.ml.start_trained_model_deployment(model_id=self.model_name)

            # define a pipeline
            self.es.ingest.put_pipeline(
                id='elser-ingest-pipeline',
                processors=[
                    {
                        'inference': {
                            'model_id': self.model_name,
                            'input_output': [
                                {
                                    'input_field': 'summary',
                                    'output_field': 'elser_embedding',
                                }
                            ]
                        }
                    }
                ]
            )
        else:
            print("To deploy elser model, you much ensure the has 'ELASTICSEARCH' as the model provider.")


    def create_index(self, index_name='my_documents', embeddings_dim=384):
        self.es.indices.delete(index=index_name, ignore_unavailable=True)
        self.es.indices.create(
            index=index_name,
            mappings={
                'properties': {
                    'embedding': {
                        'type': 'dense_vector',
                        'dims': embeddings_dim,
                        'index': True,
                        'similarity': 'cosine'
                    }
                }
            }
        )
        print(f"Index {index_name} created!")

    
    def get_embedding(self, text):
        """
        Encode a piece of text into a vector representation using the model.
        """
        return self.model.encode(text)
    
    def knn_search(self, query_text, field_to_search, field_to_return):
        """
        Conducting a simple knn search. This doesn't apply filters (yet)
        """
        query_vector = self.get_embedding(query_text) # The vector you want to search with

        # kNN search query
        knn_query={
                'query_vector': query_vector,
                'field': field_to_search,
                'k': 3,
                'num_candidates': 20,
            }

        # get a document to try search on.
        response = self.es.search(
            index="model_card_index", 
            knn=knn_query, 
            source=True
            )

        results = []
        for hit in response['hits']['hits']:
            result = {
                "model_info": hit["_source"][field_to_return]
            }
            results.append(result)
        
        return results
    
    def hybrid_search(
            self,
            query_text,
            field_to_search='full_embeddings',
            field_to_return='metadata',
            filters:dict={}
            ):
        
        # vectorize query for semantic search.
        query_vector = self.get_embedding(query_text)

        response = self.es.search(
            index="model_card_index",
            knn={
                'field': field_to_search,
                'query_vector': query_vector,
                'k': 10,
                'num_candidates': 20
            }
        )

        results = []

        for hit in response['hits']['hits']:                   
            result = hit["_source"][field_to_return]
            if filters == {}:
                results.append(result)
            elif (filters['Languages'] in result['Languages']) and (filters['task_ids'] in result['task_ids']):
                results.append(result)

        return results        
    
    def get_mapping(self, index_name='models_infos'):
        self.es.indices.get_mapping(index_name)
    
    def insert_document(self, document, passage_for_embedding, index_name='models_infos'):
        """
        :param document, dict: A document that is represented as k-v pair should be provided. For example:
        document = {
                    'title': 'Work From Home Policy',
                    'contents': 'The purpose of this full-time work-from-home policy is...',
                    'created_on': '2023-11-02',
                }
                response = es.index(index='my_documents', body=document)
                print(response['_id'])
        
        Example method for inserting documents with this method:

        ```
            import json
            from search import Search
            es = Search()
            with open('data.json', 'rt') as f:
                documents = json.loads(f.read())
            for document in documents:
                es.insert_document(document)
        ```
        """
        return self.es.index(index=index_name, document={
            **document,
            'embedding': self.get_embedding(document[passage_for_embedding])
        })
    
    def insert_documents(self, documents, passage_for_embedding, index_name='models_info'):
        """
        Inserts a batch of documents into a specified Elasticsearch index and
        adds an embedding to each document.

        This function processes a list of documents by appending a generated embedding
        to each document based on the data in the specified embedding column. Each
        document, along with its embedding, is then added to a batch of operations
        for bulk insertion into Elasticsearch.

        Parameters:
        - documents (list of dict): The list of document dictionaries to be inserted.
        - passage_for_embedding (str): The key in each document dictionary that contains the data
                            used for generating the embedding.
        - index_name (str, optional): The name of the Elasticsearch index where the
                                    documents will be inserted. Defaults to 'models_info'.

        Returns:
        - dict: The response from the Elasticsearch bulk API call, which includes
                details about the bulk operation results.

        Raises:
        - ElasticsearchException: An exception is raised if the bulk insert operation fails.

        Example usage:
        >>> insert_documents(documents=[{'id': 1, 'text': 'Sample text'}], embedding_col='text')
        {'took': 30, 'errors': False, ...}
        """
        operations = []
        for document in documents:
            operations.append({'index': {'_index': index_name}})
            operations.append({
                **document,
                'embedding': self.get_embedding(document[passage_for_embedding])
            })

        print("Bulk uploading documents")    
        return self.es.bulk(operations=operations)
        
    
    def reindex(self):
        """
        This method combines the create_index() and insert_documents() methods created earlier, 
        so that with a single call the old index can be destroyed (if it exists) and a new index built and repopulated.
        """
        self.create_index()
        with open('data.json', 'rt') as f:
            documents = json.loads(f.read())
        return self.insert_documents(documents=documents)

    def retrieve_document(self, id):
        return self.es.get(index='my_documents', id=id)
