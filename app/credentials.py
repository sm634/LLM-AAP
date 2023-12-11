import os
from dotenv import load_dotenv


def get_credentials():
    # save the credentials from .env file as environment variables.
    load_dotenv()

    # load the environment variables as a json object.
    credentials = {
        'PROJECT_ID': os.environ['PROJECT_ID'],
        'API_KEY': os.environ['API_KEY'],
        'COS_INSTANCE_CRN': os.environ['COS_INSTANCE_CRN'],
        'AUTH_ENDPOINT': os.environ['AUTH_ENDPOINT'],
        'COS_ENDPOINT': os.environ['COS_ENDPOINT'],
        'BUCKET_NAME': os.environ['BUCKET_NAME'],
        'OBJECT': os.environ['OBJECT'],
        'MODEL_ENDPOINT': os.environ['MODEL_ENDPOINT']
    }

    return credentials
