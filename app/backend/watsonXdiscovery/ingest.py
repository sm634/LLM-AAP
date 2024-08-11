import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Any, List
from elasticsearch import helpers
from sklearn.metrics.pairwise import cosine_similarity

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def get_text_embedding(texts: List[str], batch: int = 10000) -> np.ndarray:
    """
    Get the embeddings from the text.

    Args:
        texts (list(str)): List of chunks of text.
        batch (int): Batch size.
    """
    embeddings = []
    for i in range(0, len(texts), batch):
        text_batch = texts[i:(i + batch)]
        if text_batch:  # Ensure text_batch is not empty
            emb_batch = MODEL.encode(text_batch).tolist()
            embeddings.append(emb_batch)
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.array([])
    return embeddings

def create_embeddings_model(model_json):
    # Full model card embedding
    full_model_dict = model_json
    full_model = ' '.join([json.dumps(values) for values in full_model_dict.values() if values])
    full_model_embedding = get_text_embedding([full_model])

    # Model card full info embedding
    information_tags = ['long_description', 'TunningInformation', 'TrainingData', 'UsesSupported']
    full_model_information = ' '.join([json.dumps(full_model_dict[info]) for info in information_tags if info in full_model_dict])
    full_model_information_embedding = get_text_embedding([full_model_information])

    # Individual info embeddings
    def get_embedding_for_field(field):
        if field in full_model_dict and full_model_dict[field]:
            field_value = full_model_dict[field]
            if isinstance(field_value, list):  # Join lists into a single string
                field_value = ' '.join(field_value)
            return get_text_embedding([field_value])
        return np.array([])

    embedded_description = get_embedding_for_field('long_description')
    embedded_tuning_info = get_embedding_for_field('TunningInformation')
    embedded_training_data = get_embedding_for_field('TrainingData')
    embedded_uses = get_embedding_for_field('UsesSupported')
    embedded_model_overview = get_embedding_for_field('ModelOverview')

    return (
        full_model_embedding.tolist()[0] if full_model_embedding.size > 0 else [],
        full_model_information_embedding.tolist()[0] if full_model_information_embedding.size > 0 else [],
        embedded_description.tolist()[0] if embedded_description.size > 0 else [],
        embedded_tuning_info.tolist()[0] if embedded_tuning_info.size > 0 else [],
        embedded_training_data.tolist()[0] if embedded_training_data.size > 0 else [],
        embedded_uses.tolist()[0] if embedded_uses.size > 0 else [],
        embedded_model_overview.tolist()[0] if embedded_model_overview.size > 0 else []
    )

def prepare_bulk_data(documents, index_name):
    bulk_data = []
    for doc in documents:
        bulk_data.append({
            '_index': index_name,
            '_source': doc
        })
    return bulk_data

def preprocess_models(json_list, index_name_model):
    documents = []
    for model in json_list:
        full_embedding, info_embedding, desc_embedding, tuning_embedding, training_embedding, uses_embedding, overview_embedding = create_embeddings_model(model)
        
        doc = {
            'content': json.dumps(model),
            'metadata': model,  
            'full_embeddings': full_embedding,
            'information_embeddings': info_embedding,
            'description_embeddings': desc_embedding,
            'tuning_information_embeddings': tuning_embedding,
            'training_data_embeddings': training_embedding,
            'uses_supported_embeddings': uses_embedding,
            'model_overview_embeddings': overview_embedding
        }
        documents.append(doc)
    bulk_data = prepare_bulk_data(documents, index_name_model)
    return bulk_data

def ingest_models(json_list, index_name_model, es_connection):
    try:
        bulk_data = preprocess_models(json_list, index_name_model)
        if bulk_data:
            success, failed = helpers.bulk(es_connection, bulk_data, raise_on_error=False)
            return success, failed
        else:
            return 0, len(json_list)
    except Exception as e:
        raise
