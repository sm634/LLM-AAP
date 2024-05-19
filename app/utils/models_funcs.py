"""
Docstring
---------

A script that will store certain functions related to calling/instantiating LLM models and reading the relevant
templates from that LLM to be used for the chosen task.
"""
from connectors.models_connector import ModelConnector
from utils.files_handler import FileHandler


def get_model(model_provider=None, task=None):
    """
    A function to instantiate the particular instance of the model desired.
    :param model_provider: The option to change the model_provider in config, every other hyperparameter is fixed.
    :param task: The option to change the task in config, every other hyperparameter is fixed.
    :return: Foundation model of choice.
    """
    # get a particular instance of the model of choice
    if (task is not None) or (model_provider is not None):
        # This condition option give the user the chance to reassign the model that they want to use by overwriting
        # the config options for the task and model.
        file_handler = FileHandler()
        file_handler.get_config()
        config = file_handler.config

        if model_provider is not None:
            config['MODEL_PROVIDER'] = model_provider
        if task is not None:
            config['TASK'] = task

        file_handler.write_config(config)

    model_client = ModelConnector()
    model = model_client.instantiate_model()
    model_name = model_client.model_name

    return {'name': model_name,
            'model': model}
