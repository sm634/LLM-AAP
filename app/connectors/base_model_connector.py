import os

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from dotenv import load_dotenv
import yaml


class BaseModelConnector:

    def __init__(self):
        """
        BaseModelConnector reads in the config.yaml file and sets up the relevant variable values to instantiate the
        required model to be initialized using hte ModelConnector class, handling dependencies for model access and
        inference.
        """
        # we will need the environment variables for credentials.
        global provider_task
        load_dotenv()

        # instantiate the credentials values
        self.api_key = ''
        self.project_id = ''
        self.model_endpoint = ''

        try:
            # instantiate the configs with relative to example_main.py script.
            with open('configs/config.yaml', 'r') as file:
                self.config = yaml.safe_load(file)

        except FileNotFoundError:
            # instantiate based on the path provided from .env file.
            config_path = os.environ['CONFIG_PATH']
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)

        # get the model provider and the task of interest.
        self.model_provider = self.config['MODEL_PROVIDER'].lower()
        self.task = self.config['TASK'].lower()

        if self.model_provider == 'openai':
            # get the credentials
            self.api_key = os.environ['OPENAI_API_KEY']

            if self.task == 'article_classifier':
                provider_task = self.config['OPENAI']['ARTICLE_CLASSIFIER']

        elif self.model_provider == 'watsonx':
            # get the watsonx credentials
            self.api_key = os.environ['WATSONX_API_KEY']
            self.project_id = os.environ['PROJECT_ID']
            self.model_endpoint = os.environ['MODEL_ENDPOINT']

            if self.task == 'article_classifier':
                provider_task = self.config['WATSONX']['ARTICLE_CLASSIFIER']
        else:
            raise

        model_type = provider_task['model_type']
        if not isinstance(model_type, str):
            self.model_type = getattr(ModelTypes, model_type)
            self.model_name = self.model_type.name
        else:
            self.model_type = model_type
            self.model_name = model_type

        # decoding method
        decoding_method = provider_task['decoding_method']
        if not isinstance(decoding_method, str):
            self.decoding_method = getattr(DecodingMethods, decoding_method)
        else:
            self.decoding_method = decoding_method

        # set the hyperparameters according to the values in the config file.
        self.max_tokens = provider_task['max_tokens']
        self.min_tokens = provider_task['min_tokens']
        self.temperature = provider_task['temperature']

        self.params = {
            GenParams.MAX_NEW_TOKENS: self.max_tokens,
            GenParams.MIN_NEW_TOKENS: self.min_tokens,
            GenParams.DECODING_METHOD: self.decoding_method,
            GenParams.TEMPERATURE: self.temperature
        }
