import os
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


class CONFIG:
    IBM_CLOUD_URL = os.getenv("IBM_CLOUD_URL")
    API_KEY = os.getenv("API_KEY")
    PROJECT_ID = os.getenv("PROJECT_ID")

    COS_RESOURCE_INSTANCE_ID =  os.getenv("COS_RESOURCE_INSTANCE_ID")
    COS_ENDPOINT_URL = os.getenv("COS_ENDPOINT_URL")

    COS_API_KEY = os.getenv("COS_API_KEY")
    COS_BUCKET = os.getenv("COS_BUCKET")

    COS_HMAC_ACCESS_KEY_ID = os.getenv("COS_HMAC_ACCESS_KEY_ID")
    COS_HMAC_SECRET_ACCESS_KEY = os.getenv("COS_HMAC_SECRET_ACCESS_KEY")

    ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
    ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
    ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
    ELASTICSEARCH_MODEL_INDEX_NAME =  os.getenv("ELASTICSEARCH_MODEL_INDEX_NAME")
