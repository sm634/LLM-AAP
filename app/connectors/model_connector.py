from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

import yaml


class WatsonXModel:

    def __init__(self, api_key, model_endpoint, project_id, config_path):
        """
        The Watson Model class initializing parameters.
        :param api_key: valid api key to access the model
        :param model_endpoint: the url to access the model
        :param project_id: the watsonx project id
        :param config_path: the path to the config file with hyperparameters
        """
        self.API_KEY = api_key
        self.ENDPOINT = model_endpoint
        self.PROJECT_ID = project_id

        # instantiate the configs
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # model
        model_type = config['MODEL_HYPERPARAMETERS']['model_type']
        self.model_type = getattr(ModelTypes, model_type)

        # decoding method
        decoding_method = config['MODEL_HYPERPARAMETERS']['decoding_method']
        self.decoding_method = getattr(DecodingMethods, decoding_method)

        # set the hyperparameters according to the values in the config file.
        self.max_tokens = config['MODEL_HYPERPARAMETERS']['max_tokens']
        self.min_tokens = config['MODEL_HYPERPARAMETERS']['min_tokens']
        self.temperature = config['MODEL_HYPERPARAMETERS']['temperature']

        self.params = {
            GenParams.MAX_NEW_TOKENS: self.max_tokens,
            GenParams.MIN_NEW_TOKENS: self.min_tokens,
            GenParams.DECODING_METHOD: self.decoding_method,
            GenParams.TEMPERATURE: self.temperature
        }

    def set_params(self, max_tokens, min_tokens, decoding_method, temperature, model_type):
        """A method used to override the hyperparameters, if needed."""
        # reassign the attribute values for hyperparameters.
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.decoding_method = decoding_method
        self.temperature = temperature
        self.model_type = model_type

        self.params = {
            GenParams.MAX_NEW_TOKENS: max_tokens,
            GenParams.MIN_NEW_TOKENS: min_tokens,
            GenParams.DECODING_METHOD: decoding_method,
            GenParams.TEMPERATURE: temperature
        }

    def instantiate_model(self):

        model = Model(
            model_id=self.model_type,
            params=self.params,
            credentials={
                "apikey": self.API_KEY,
                "url": self.ENDPOINT
            },
            project_id=self.PROJECT_ID
        )
        return model
