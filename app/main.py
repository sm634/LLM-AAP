from credentials import get_credentials
from connectors.model_connector import WatsonModel

credentials = get_credentials()
api_key = credentials['API_KEY']
model_endpoint = credentials['MODEL_ENDPOINT']
project_id = credentials['PROJECT_ID']

model = WatsonModel(api_key=api_key,
                    model_endpoint=model_endpoint,
                    watsonx_project_id=project_id,
                    config_path='configs/hyperparameters.yaml')

print(model)
breakpoint()
