from utils.webcrawl import get_webpage_content
from utils.create_embeddings import chunk_texts_list, MiniLML6V2EmbeddingFunction
from connectors.vector_db_connector import MilvusConnector


def create_knowledge_base(
        db='Milvus',
        collection_name='watsonx_gov_collection_urls',
        subject='Watsonx.gov Documents with urls',
        websites_file='watsonx.gov.txt',
        custom_schema=True
    ):
    """
    A function that creates a knowledge base in Milvus by:
        i. Scraping websites as data source.
        ii. Chunking them into separate smaller documents.
        iii. Using an embedding model to vectorize the documents.
        iv. Inserting those embeddings and documents to the vector database collection.
    """
    soup_text_url = get_webpage_content(file_name=websites_file)
    url_documents = chunk_texts_list(soup_text_url) # this returns a List[Tuple]. The tuple contains (url, text_chunk)
    urls = [i[0] for i in url_documents]
    documents = [i[1] for i in url_documents]
    print("Chunked Documents created")

    # create embeddings
    print("Creating embeddings from documents")
    embeddings = MiniLML6V2EmbeddingFunction(documents)
    n_embeddings = len(embeddings)
    print("Embeddings Created")
    print("No. of Embeddings ", n_embeddings)

    if db == 'Milvus':

        # instantiate vector db
        milvus_conn = MilvusConnector()

        COLLECTION_NAME = collection_name

        if custom_schema:
            # create the custom schema for the Milvus Collection.
            milvus_conn.create_doc_url_schema() # includes, embeddings, text, url and id as fields.
            # prepare data with embeddings to upload.
            data = [
                {
                    "id": i, 
                    "embeddings": embeddings[i], 
                    "text": str(documents[i]),
                    "url": str(urls[i]),
                    "subject": subject
                }
                for i in range(n_embeddings)
            ]
            
        else:
            
            data = [
                {
                    "id": i,
                    "vector": embeddings[i],
                    "text": str(documents[i]),
                    "subject": subject
                }
                for i in range(n_embeddings)
            ]

        milvus_conn.create_collection(collection_name=COLLECTION_NAME)
        print("Milvus Collection Created")


        print(f"Inserting Data to Milvus collection: {COLLECTION_NAME}")
        result = milvus_conn.insert(
            collection_name=COLLECTION_NAME,
            data=data
            )
        print(f"Data inserted to Milvus collection: {COLLECTION_NAME}")
        
        return result
    
    else:
        return "Select a supported Search Database. Example 'Milvus' as an argument."
