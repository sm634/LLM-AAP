from credentials import get_credentials

from connectors.model_connector import WatsonXModel

from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from utils.files_handler import FileHandler


def prepare_credentials():
    # get credentials
    credentials = get_credentials()
    credentials = {
        'api_key': credentials['API_KEY'],
        'model_endpoint': credentials['MODEL_ENDPOINT'],
        'project_id': credentials['PROJECT_ID']
    }
    return credentials


def get_model(api_key,
              model_endpoint,
              project_id):
    """
    A function to instantiate the particular instance of the model desired.
    :param api_key: the appropriate api key to connect to Watsonx.ai foundation model.
    :param model_endpoint: tthe url used to access the model.
    :param project_id: Watsonx.ai project id containing the assets and resources of interest.
    :param model_type: The type of model. By default is imported as MODEL_TYPE from the model_select module.
    :return: Watsonx.ai Foundation model of choice.
    """

    # initialize the WatsonxModel class with the appropriate model resource, project and hyperparameters.
    model_client = WatsonXModel(api_key=api_key,
                                model_endpoint=model_endpoint,
                                project_id=project_id,
                                config_path='configs/hyperparameters.yaml')
    # get a particular instance of the model of choice
    model = model_client.instantiate_model()

    return model


def get_prompt_template(file_name):
    """
    Get prompt template from file.
    :param file_name: the name of the file for the prompt template, including the .txt extension.
    :return: prompt str
    """
    file_handler = FileHandler()
    file_handler.get_prompt_from_file(file_name)
    prompt_template = file_handler.prompt
    return prompt_template


def get_data_df(file_name):
    """
    Get data dataframe from file.
    :param file_name: the name of the file for the data to move to DataFrame.
    :return: pandas DataFrame
    """
    file_handler = FileHandler()
    file_handler.get_data_from_file(file_name=file_name)
    df = file_handler.data
    return df


def run_article_classifier():
    """
    Run the entire pipeline E2E.
    """
    # prepare the credentials
    credentials = prepare_credentials()

    # get the model of choice
    base_model = get_model(
        credentials['api_key'],
        credentials['model_endpoint'],
        credentials['project_id']
    )

    # integrate with langchain Watsonx LLM model
    langchain_model = WatsonxLLM(model=base_model)

    # get the prompt template
    red_flag_template = get_prompt_template(file_name='classify_article.txt')
    prompt_template = PromptTemplate.from_template(red_flag_template)

    # get the data
    df = get_data_df(file_name='First200_ic.csv')
    sample_article = df.sample(1)['article']

    # specify the input variables on the prompt template to create the prompt.
    prompt_inputs = {'article': sample_article}

    llm_chain = LLMChain(prompt=prompt_template, llm=langchain_model)

    response_text = llm_chain.run(prompt_inputs)
    print(response_text)
    return response_text

