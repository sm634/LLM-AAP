import xml.etree.ElementTree as ET
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Any
from elasticsearch import helpers
from sklearn.metrics.pairwise import cosine_similarity

# MODEL = SentenceTransformer("all-MiniLM-L6-v2")

#  ***************** to fetch the vector embeddings of a JD *****************
def fetch_embedding_jd_embeddings(index_name, jd_param, es_connection):
    """
    Fetch the embeddings of a specific document from an index.
    
    :param index_name: Name of the Elasticsearch index.
    :param document_id: ID of the document to fetch.
    :return: Embedding vector of the document.
    """

    query = {
        "query":{
            "query_string":{
                # "query": "290"
                "query": jd_param
            }
        }
    }
    response = es_connection.search(index=index_name, body=query)

    results = []
    for hit in response['hits']['hits']:
        result = {
            "jd-id": hit["_source"]["metadata"]['id'],
            "overall_embeddings": hit["_source"]['overall_embeddings'],
            "job-history_embeddings": hit["_source"]["job-description_embeddings"], # changing variable name to job-history_embeddings to accomodate search for CVs
            "responsibilities_embeddings": hit["_source"]['responsibilities_embeddings'],
            "education_embeddings": hit["_source"]['education_embeddings'],
            "skills_embeddings": hit["_source"]['skills_embeddings']
        }
        results = result
    return results
#  ***************** to fetch the vector embeddings of a JD *****************


#  ***************** to fetch the vector embeddings of a CV *****************
def fetch_embedding_cv_embeddings(index_name, cv_param, es_connection):
    """
    Fetch the embeddings of a specific document from an index.
    
    :param index_name: Name of the Elasticsearch index.
    :param document_id: ID of the document to fetch.
    :return: Embedding vector of the document.
    """

    query = {
        "query":{
            "query_string":{
                "query": cv_param
            }
        }
    }
    response = es_connection.search(index=index_name,body=query)
    results = []
    for hit in response['hits']['hits']:
        result = {
            "jd-id": hit["_source"]["metadata"]['JD_id'],
            "resume-id": hit['_source']['metadata']['resume_id'],
            "overall_embeddings": hit["_source"]['overall_embeddings'],
            "job-description_embeddings": hit["_source"]['job-history_embeddings'], # changing variable name to job-description_embeddings to accomodate search for JDs
            "responsibilities_embeddings": hit["_source"]['responsibilities_embeddings'],
            "education_embeddings": hit["_source"]['education_embeddings'],
            "skills_embeddings": hit['_source']['skills_embeddings']
        }
        results = result
    return results
#  ***************** to fetch the vector embeddings of a JD *****************


# ***************** To search and get a match score for CVs related to a specific JD *****************
def multi_level_search_cv_withJD(index_name, metadata_field, metadata_value, query_embedding, top_k, threshold, es_connection):
    query = {
        "size": top_k,
        "_source": ["content", "metadata", "embeddings"],  # Specify fields to retrieve
        "query": {
            "bool": {
                "should": [
                    {
                        "script_score": {
                            "query" : {
                                "bool" : {
                                "filter" : {
                                    "term" : {
                                    f"metadata.{metadata_field}" : metadata_value
                                    }
                                }
                                }
                            },
                            "min_score": 1 + threshold,
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'overall_embeddings') + 1.0",
                                "params": {"query_vector": query_embedding}
                            }
                        }
                    }
                ]
            }
        }
    }
    
    response = es_connection.search(index=index_name, body=query)

    results = []
    for hit in response['hits']['hits']:
        result = {
            "content": hit["_source"]["content"],
            "jd-id": hit["_source"]["metadata"]['JD_id'],
            "resume-id": hit["_source"]["metadata"]['resume_id'],
            "score": hit['_score']-1
        }
        results.append(result)
    
    return results

# ***************** To search and get a match score for CVs related to a specific JD based on categories *****************
def multi_level_search_cv_withJD_category(index_name, metadata_field, metadata_value, query_embedding, top_k, threshold, es_connection):
    embedding_fields = ['job-history_embeddings','responsibilities_embeddings', 'education_embeddings','skills_embeddings']  # Add your embedding field names here
    results = []
    
    for field in embedding_fields:
        # Create a script_score query for each embedding
        script_source = f"cosineSimilarity(params.query_vector, '{field}') + 1.0"
        query_vector_param = {"query_vector": query_embedding[field]}
        
        query = {
            "size": top_k,  # Limit the number of results to top_k
            "_source": ["content", "metadata", "overall_embeddings", "job-history_embeddings", "responsibilities_embeddings", "education_embeddings", "skills_embeddings"],  # Specify fields to retrieve
            "query": {
                "bool": {
                    "should": [
                        {
                            "script_score": {
                                "query" : {
                                    "bool" : {
                                    "filter" : {
                                        "term" : {
                                            f"metadata.{metadata_field}" : metadata_value
                                        }
                                    }
                                    }
                                },
                                "min_score": 1 + threshold,
                                "script": {
                                    "source": script_source,
                                    "params": query_vector_param
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        response = es_connection.search(index=index_name, body=query)

        embedding_results = []
        for hit in response['hits']['hits']:
            result = {
                "content": hit["_source"]["content"],
                "jd-id": hit["_source"]["metadata"]['JD_id'],
                "resume-id": hit["_source"]["metadata"]['resume_id'],
                "job-history-embeddings": hit["_source"]["job-history_embeddings"],
                "responsibilities_embeddings": hit["_source"]['responsibilities_embeddings'],
                "education_embeddings": hit["_source"]['education_embeddings'],
                "skills_embeddings": hit['_source']['skills_embeddings']
            }
            embedding_results.append(result)

        results.append({
                "embedding_field": field,
                "results": embedding_results
            })
        
    similarity_scores = []
    for cat_embeddings in results:
        for top_match in cat_embeddings['results']:
            history_score = cosine_similarity([query_embedding['job-history_embeddings']], [top_match['job-history-embeddings']])
            responsibilities_score = cosine_similarity([query_embedding['responsibilities_embeddings']], [top_match['responsibilities_embeddings']])
            skills_score = cosine_similarity([query_embedding['skills_embeddings']], [top_match['skills_embeddings']])
            education_score = cosine_similarity([query_embedding['education_embeddings']], [top_match['education_embeddings']])
            average_score = np.mean([history_score,responsibilities_score,skills_score,education_score])
            # print(average_score)
            if average_score > threshold:
                similarity_scores.append({
                    'content': top_match['content'],
                    'jd-id': query_embedding['jd-id'],
                    'resume-id': top_match['resume-id'],
                    'score': average_score
                })
    
    sorted_data = sorted(similarity_scores, key=lambda x: x['score'], reverse=True)
    # print(sorted_data)

    # Use a set to track seen document ID pairs
    seen = set()
    unique_sorted_data = []

    for item in sorted_data:
        doc_pair = (item['jd-id'], item['resume-id'])
        if doc_pair not in seen:
            seen.add(doc_pair)
            unique_sorted_data.append(item)

    if top_k == 10000:
        return unique_sorted_data[:len(unique_sorted_data)]
    else:
        return unique_sorted_data[:top_k]
# ***************** To search and get a match score for CVs related to a specific JD based on categories *****************

# to search all the CVs to find top match for JD
def search_jds_overall(index_name, query_embedding, top_k, threshold, es_connection):
    query = {
        "size": top_k,
        "_source": ["content", "metadata", "overall_embeddings"],  # Specify fields to retrieve
        "query": {
            "bool": {
                "should": [
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "min_score": 1 + threshold,
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'overall_embeddings')+1.0",
                                "params": {"query_vector": query_embedding['overall_embeddings']}
                            }
                        }
                    }
                ]
            }
        }
    }
    
    response = es_connection.search(index=index_name, body=query)

    results = []
    for hit in response['hits']['hits']:
        result = {
            "content": hit["_source"]["content"],
            "jd-id": hit["_source"]["metadata"]['id'],
            "score": hit['_score']-1
        }
        results.append(result)
    return results
    

def search_jds_category(index_name, query_embedding, top_k, threshold, es_connection):
    embedding_fields = ['job-description_embeddings', 'responsibilities_embeddings', 'education_embeddings','skills_embeddings']  # Add your embedding field names here
    results = []
    
    for field in embedding_fields:
        # Create a script_score query for each embedding
        script_source = f"(cosineSimilarity(params.query_vector, '{field}') + 1.0)"
        query_vector_param = {"query_vector": query_embedding[field]}
        
        query = {
            "size": top_k,  # Limit the number of results to top_k
            "_source": ["content", "metadata", "overall_embeddings", "job-description_embeddings", "responsibilities_embeddings", "education_embeddings", "skills_embeddings"],  # Specify fields to retrieve
            "query": {
                "bool": {
                    "must": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "min_score": 1 + threshold,
                                "script": {
                                    "source": script_source,
                                    "params": query_vector_param
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        response = es_connection.search(index=index_name, body=query)

        embedding_results = []
        for hit in response['hits']['hits']:
            result = {
                "content": hit["_source"]["content"],
                "jd-id": hit["_source"]["metadata"]['id'],
                "score": hit["_score"],
                "job-description_embeddings": hit["_source"]["job-description_embeddings"],
                "responsibilities_embeddings": hit["_source"]['responsibilities_embeddings'],
                "education_embeddings": hit["_source"]['education_embeddings'],
                "skills_embeddings": hit['_source']['skills_embeddings']
                # "metadata": hit["_source"]["metadata"]
            }
            embedding_results.append(result)

        results.append({
                "embedding_field": field,
                "results": embedding_results
            })
    
    similarity_scores = []
    for cat_embeddings in results:
        for top_match in cat_embeddings['results']:
            history_score = cosine_similarity([query_embedding['job-description_embeddings']], [top_match['job-description_embeddings']])
            responsibilities_score = cosine_similarity([query_embedding['responsibilities_embeddings']], [top_match['responsibilities_embeddings']])
            skills_score = cosine_similarity([query_embedding['skills_embeddings']], [top_match['skills_embeddings']])
            education_score = cosine_similarity([query_embedding['education_embeddings']], [top_match['education_embeddings']])
            average_score = np.mean([history_score,responsibilities_score,skills_score,education_score])
            # print(average_score)
            if average_score > threshold:
                similarity_scores.append({
                        'content': top_match['content'],
                        'jd-id': top_match['jd-id'],
                        'score': average_score
                })

    sorted_data = sorted(similarity_scores, key=lambda x: x['score'], reverse=True)

    # Use a set to track seen document ID pairs
    seen = set()
    unique_sorted_data = []

    for item in sorted_data:
        doc_pair = (item['jd-id'])
        if doc_pair not in seen:
            seen.add(doc_pair)
            unique_sorted_data.append(item)

    if top_k == 10000:
        return unique_sorted_data[:len(unique_sorted_data)]
    else:
        return unique_sorted_data[:top_k]
    

# ************************* To search all the cvs that match with the JD (128) ***************************************
# overall embeddings match
def search_cvs_overall(index_name, query_embedding, top_k, threshold, es_connection):
    print("top_k", top_k, "threshold:", threshold)
    query = {
        "size": top_k,
        "_source": ["content", "metadata", "overall_embeddings"],  # Specify fields to retrieve
        "query": {
            "bool": {
                "should": [
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "min_score": 1 + threshold,
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'overall_embeddings') + 1.0",
                                "params": {"query_vector": query_embedding['overall_embeddings']}
                            }
                        }
                    }
                ]
            }
        }
    }
    
    response = es_connection.search(index=index_name, body=query)

    results = []
    for hit in response['hits']['hits']:
        result = {
            "content": hit["_source"]["content"],
            # "jd-id": hit["_source"]["metadata"]['JD_id'],
            "resume-id": hit["_source"]["metadata"]['resume_id'],
            "candidate-id": hit['_source']['metadata']['candidate_id'],
            "score": hit['_score']-1
        }
        results.append(result)
    
    return results

# category wise embeddings match
def search_cvs_category(index_name, query_embedding, top_k, threshold, es_connection):
    embedding_fields = ['job-history_embeddings','responsibilities_embeddings', 'education_embeddings','skills_embeddings']  # Add your embedding field names here
    results = []
    
    for field in embedding_fields:
        # Create a script_score query for each embedding
        script_source = f"(cosineSimilarity(params.query_vector, '{field}')+1.0)" 
        query_vector_param = {"query_vector": query_embedding[field]}
        
        query = {   
            "size": top_k,  # Limit the number of results to top_k
            "_source": ["content", "metadata", "overall_embeddings", "job-history_embeddings", "responsibilities_embeddings", "education_embeddings", "skills_embeddings"],  # Specify fields to retrieve
            "query": {
                "bool": {
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "min_score": 1 + threshold,
                                "script": {
                                    "source": script_source,
                                    "params": query_vector_param
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        response = es_connection.search(index=index_name, body=query)

        embedding_results = []
        # if response['hits']['total']['value']:
        for hit in response['hits']['hits']:
            result = {
                "content": hit["_source"]["content"],
                "jd-id": hit["_source"]["metadata"]['JD_id'],
                "resume-id": hit["_source"]["metadata"]['resume_id'],
                "candidate-id": hit["_source"]['metadata']['candidate_id'],
                "job-history-embeddings": hit["_source"]["job-history_embeddings"],
                "responsibilities_embeddings": hit["_source"]['responsibilities_embeddings'],
                "education_embeddings": hit["_source"]['education_embeddings'],
                "skills_embeddings": hit['_source']['skills_embeddings']
            }
            embedding_results.append(result)

        results.append({
                "embedding_field": field,
                "results": embedding_results
            })

    similarity_scores = []
    for cat_embeddings in results:
        for top_match in cat_embeddings['results']:
            history_score = cosine_similarity([query_embedding['job-history_embeddings']], [top_match['job-history-embeddings']])
            responsibilities_score = cosine_similarity([query_embedding['responsibilities_embeddings']], [top_match['responsibilities_embeddings']])
            skills_score = cosine_similarity([query_embedding['skills_embeddings']], [top_match['skills_embeddings']])
            education_score = cosine_similarity([query_embedding['education_embeddings']], [top_match['education_embeddings']])
            average_score = np.mean([history_score,responsibilities_score,skills_score,education_score])
            # print(average_score)
            if average_score > threshold:
                similarity_scores.append({
                    'content': top_match['content'],
                    'resume-id': top_match['resume-id'],
                    'candidate-id': top_match['candidate-id'],
                    'score': average_score
                })
    
    sorted_data = sorted(similarity_scores, key=lambda x: x['score'], reverse=True)
    # print(sorted_data)

    # Use a set to track seen document ID pairs
    seen = set()
    unique_sorted_data = []

    for item in sorted_data:
        doc_pair = (item['resume-id'], item['candidate-id'])
        if doc_pair not in seen:
            seen.add(doc_pair)
            unique_sorted_data.append(item)

    if top_k == 10000:
        return unique_sorted_data[:len(unique_sorted_data)]
    else:
        return unique_sorted_data[:top_k]
    

def search_jd_combined_results(overall_embeddings, category_embeddings, top_k):
     # Create dictionaries to map id1 to their respective entries
    oe_list = {entry['jd-id']: entry for entry in overall_embeddings}
    ce_list = {entry['jd-id']: entry for entry in category_embeddings}

    # Find the intersection of id1
    intersection_ids = set(oe_list.keys()) & set(ce_list.keys())
    print(intersection_ids)

    # Compute the average score for the entries with the same id1
    intersection_list = []
    for jd_id in intersection_ids:
        entry1 = oe_list[jd_id]
        entry2 = ce_list[jd_id]
        average_score = (entry1['score'] + entry2['score']) / 2
        # Create a new entry with the average score
        new_entry = entry1.copy()  # Copy one of the entries to preserve id2 and id3
        new_entry['score'] = average_score
        intersection_list.append(new_entry)

    # Sort the list based on score
    sorted_intersection_list = sorted(intersection_list, key=lambda x: x['score'], reverse=True)

    if top_k == 10000:
        return sorted_intersection_list[:len(sorted_intersection_list)]
    else:
        if len(sorted_intersection_list) < top_k:
            return sorted_intersection_list[:len(sorted_intersection_list)]
        else:
            return sorted_intersection_list[:top_k]



def search_cv_combined_results(overall_embeddings, category_embeddings, top_k):

    # Create dictionaries to map id1 to their respective entries
    oe_list = {entry['candidate-id']: entry for entry in overall_embeddings}
    ce_list = {entry['candidate-id']: entry for entry in category_embeddings}

    # Find the intersection of id1
    intersection_ids = set(oe_list.keys()) & set(ce_list.keys())
    print(intersection_ids)

    # Compute the average score for the entries with the same id1
    intersection_list = []
    for candidate_id in intersection_ids:
        entry1 = oe_list[candidate_id]
        entry2 = ce_list[candidate_id]
        average_score = (entry1['score'] + entry2['score']) / 2
        # Create a new entry with the average score
        new_entry = entry1.copy()  # Copy one of the entries to preserve id2 and id3
        new_entry['score'] = average_score
        intersection_list.append(new_entry)

    # Sort the list based on score
    sorted_intersection_list = sorted(intersection_list, key=lambda x: x['score'], reverse=True)

    if top_k == 10000:
        return sorted_intersection_list[:len(sorted_intersection_list)]
    else:
        if len(sorted_intersection_list) < top_k:
            return sorted_intersection_list[:len(sorted_intersection_list)]
        else:
            return sorted_intersection_list[:top_k]