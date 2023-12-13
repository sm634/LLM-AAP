import os

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

from langchain.chat_models import ChatOpenAI

import yaml


class FoundationModel:

    def __init__(self,
                 model_provider,
                 api_key,
                 config_path,
                 model_endpoint=None,
                 project_id=None,
                 ):
        """
        The Foundation Model class initializing parameters, many can be used/is relevant for different model providers.
        :param model_provider: The foundation model provider from: ['watsonx', 'openai']
        :param api_key: valid api key to access the model
        :param model_endpoint: the url to access the model
        :param project_id: the watsonx project id, if applicable
        :param config_path: the path to the config file with hyperparameters
        """
        # The model provider will be from a list of model providers.
        self.model_provider = model_provider.lower()
        assert self.model_provider in ['watsonx', 'openai'], ("Please choose one of the eligible providers: ["
                                                              "'watsonx', 'openai']")

        # credentials details to access and connect to the model
        self.API_KEY = api_key
        self.ENDPOINT = model_endpoint
        self.PROJECT_ID = project_id

        # instantiate the configs
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # model
        model_type = config['MODEL_HYPERPARAMETERS']['model_type']
        if not isinstance(model_type, str):
            self.model_type = getattr(ModelTypes, model_type)
            self.model_name = self.model_type.name
        else:
            self.model_type = model_type
            self.model_name = model_type

        # decoding methodc
        decoding_method = config['MODEL_HYPERPARAMETERS']['decoding_method']
        if not isinstance(decoding_method, str):
            self.decoding_method = getattr(DecodingMethods, decoding_method)
        else:
            self.decoding_method = decoding_method

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

        if self.model_provider == 'watsonx':
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

        elif self.model_provider == 'openai':

            model = ChatOpenAI(
                api_key=self.API_KEY,
                temperature=self.temperature,
                model=self.model_type
            )

            return model

        else:
            pass
