"""
Docstring
---------

The base class to connect to models. There are currently supported for LLM foundation models from OpenAI and Watsonx.
The particular values assigned to parameters specified within this class will depend on the config file. Once the
values have been assigned to the relevant class attributes, these will be inherited in ModelsConnector class.
"""
import os

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

from dotenv import load_dotenv
from utils.files_handler import FileHandler


class BaseModelConnector:

    def __init__(self):
        """
        BaseModelConnector reads in the models_config.yaml file and sets up the relevant variable values to instantiate the
        required model to be initialized using hte ModelConnector class, handling dependencies for model access and
        inference.
        """
        # we will need the environment variables for credentials.
        self.provider_task = None
        load_dotenv()

        # instantiate the credentials values
        self.api_key = ''
        self.project_id = ''
        self.model_endpoint = ''

        # get models configs
        file_handler = FileHandler()
        file_handler.get_config('models_config.yaml')
        self.config = file_handler.config

        # get the model provider and the task of interest.
        self.model_provider = self.config['MODEL_PROVIDER'].lower()
        self.task = self.config['TASK'].lower()

        if self.model_provider == 'openai':
            # get the credentials
            self.api_key = os.environ['OPENAI_API_KEY']

            if self.task == 'text_classifier':
                self.provider_task = self.config['OPENAI']['TEXT_CLASSIFIER']
            elif self.task == 'preprocess_article':
                self.provider_task = self.config['OPENAI']['PREPROCESS_ARTICLE']
            elif self.task == 'text_comparator':
                self.provider_task = self.config['OPENAI']['TEXT_COMPARATOR']
            elif self.task == 'embeddings_comparator':
                self.provider_task = self.config['OPENAI']['EMBEDDINGS_COMPARATOR']
            elif self.task == 'redflag_article_comparator':
                self.provider_task = self.config['OPENAI']['REDFLAG_ARTICLE_COMPARATOR']
            elif self.task == 'extract_fields':
                self.provider_task = self.config['OPENAI']['EXTRACT_FIELDS']
            elif self.task == 'summarizer':
                self.provider_task = self.config['OPENAI']['SUMMARIZER']
            elif self.task == 'sentiment_parser':
                self.provider_task = self.config['OPENAI']['COMPLAINTS_PARSER']

        elif self.model_provider == 'watsonx':
            # get the watsonx credentials
            self.api_key = os.environ['WATSONX_API_KEY']
            self.project_id = os.environ['PROJECT_ID']
            self.model_endpoint = os.environ['MODEL_ENDPOINT']

            if self.task == 'text_classifier':
                self.provider_task = self.config['WATSONX']['TEXT_CLASSIFIER']
            elif self.task == 'preprocess_article':
                self.provider_task = self.config['WATSONX']['PREPROCESS_ARTICLE']
            elif self.task == 'text_comparator':
                self.provider_task = self.config['WATSONX']['TEXT_COMPARATOR']
            elif self.task == 'embeddings_comparator':
                self.provider_task = self.config['WATSONX']['EMBEDDINGS_COMPARATOR']
            elif self.task == 'redflag_article_comparator':
                self.provider_task = self.config['WATSONX']['REDFLAG_ARTICLE_COMPARATOR']
            elif self.task == 'extract_fields':
                self.provider_task = self.config['WATSONX']['EXTRACT_FIELDS']
            elif self.task == 'summarizer':
                self.provider_task = self.config['WATSONX']['SUMMARIZER']
            elif self.task == 'complaints_parser':
                self.provider_task = self.config['WATSONX']['COMPLAINTS_PARSER']
        else:
            raise

        model_type = self.provider_task['model_type']
        if self.model_provider == 'watsonx':
            self.model_type = getattr(ModelTypes, model_type)
            self.model_name = self.model_type.name
        else:
            self.model_type = model_type
            self.model_name = model_type

        # decoding method
        decoding_method = self.provider_task['decoding_method']
        if self.model_provider == 'watsonx':
            self.decoding_method = getattr(DecodingMethods, decoding_method)
        else:
            self.decoding_method = decoding_method

        # set the hyperparameters according to the values in the config file.
        self.max_tokens = self.provider_task['max_tokens']
        self.min_tokens = self.provider_task['min_tokens']
        self.temperature = self.provider_task['temperature']
        self.top_p = self.provider_task['top_p']
        self.top_k = self.provider_task['top_k']
        self.repetition_penalty = self.provider_task['repetition_penalty']
