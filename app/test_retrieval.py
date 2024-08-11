from connectors.elasticsearch_connector import ElasticSearchConnector
from connectors.vector_db_connector import MilvusConnector
from utils.files_handler import FileHandler

file_handler = FileHandler()
file_handler.get_config('vector_db_config.yaml')
VECTOR_STORE = file_handler.config['VECTOR_STORE']


def test_elasticsearch_retrieval():
    esc = ElasticSearchConnector()

    query = "I need a model that can work on the japanese language."
    query_vector = esc.get_embedding(query)
    knn= {
            'query_vector': query_vector,
            'field': 'embedding',
            'k': 3,
            'num_candidates': 20
        }

    response = esc.es.search(index='models_infos', knn=knn, source=True)
    retrieved_documents_data = response['hits']['hits']

    documents = []
    models = []
    for i, data in enumerate(retrieved_documents_data):
        documents.append(response['hits']['hits'][i]['_source']['document'])
        models.append(response['hits']['hits'][i]['_source']['model_info']['Model'])

    breakpoint()


def test_milvus_retrieval():
    # instantiate vector db
    milvus_conn = MilvusConnector()

    COLLECTION_NAME = 'test_collection'

    test_questions = ["What is Watsonx.governance?"]

    print("Searching Knowledge Base")
    retrieved_results = []
    for question in test_questions:
        result = milvus_conn.search(
            collection_name=COLLECTION_NAME,
            query=question,
            anns_field='embeddings',
            fields=["text", "url", "subject"]
        )
        retrieved_results.append(result)


if VECTOR_STORE == 'MILVUS':
    test_milvus_retrieval()
elif VECTOR_STORE == 'ELASTICSEARCH':
    test_elasticsearch_retrieval()