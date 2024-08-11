from connectors.elasticsearch_connector import ElasticSearchConnector
# from watsonXdiscovery import watsonXdiscovery_properties as wXdProp

client = ElasticSearchConnector()
es = client.es

print("\nRetrieving Search Results\n")

query_text = """
I would like to use a model for a text classification task. Ideally this will be a small model that is easy to fine-tune.
The use case focuses on ensuring that we can extract from a user query certain key words associated with a job service,
and classify it according to the industry that this would fit into.
"""

filters = {
    'Languages': 'Telugu',
    'task_ids': 'classification'
}

# without filter
results_1 = client.hybrid_search(
    query_text=query_text,
    field_to_search='full_embeddings',
    field_to_return='metadata'
)

# with filter
results_2 = client.hybrid_search(
    query_text=query_text,
    field_to_search='full_embeddings',
    field_to_return='metadata',
    filters=filters
)

breakpoint()
