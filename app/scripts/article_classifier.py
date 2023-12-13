from credentials import get_credentials

from connectors.model_connector import FoundationModel

from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from utils.files_handler import FileHandler

# global variables used in local functions
file_handler = FileHandler()


def prepare_credentials():
    # get credentials
    credentials = get_credentials()
    credentials = {
        'watsonx_api_key': credentials['WATSONX_API_KEY'],
        'model_endpoint': credentials['MODEL_ENDPOINT'],
        'project_id': credentials['PROJECT_ID'],
        'openai_api_key': credentials['OPENAI_API_KEY']
    }
    return credentials


def get_model(model_provider,
              api_key,
              model_endpoint=None,
              project_id=None):
    """
    A function to instantiate the particular instance of the model desired.
    :param model_provider: either 'watsonx' or 'openai'
    :param api_key: the appropriate api key to connect to Watsonx.ai foundation model.
    :param model_endpoint: tthe url used to access the model.
    :param project_id: Watsonx.ai project id containing the assets and resources of interest.
    :param model_type: The type of model. By default is imported as MODEL_TYPE from the model_select module.
    :return: Watsonx.ai Foundation model of choice.
    """

    model_client = ''

    if model_provider.lower() == 'watsonx':
        # initialize the Watsonx model class with the appropriate model resource, project and hyperparameters.
        model_client = FoundationModel(model_provider=model_provider,
                                       api_key=api_key,
                                       config_path='configs/watsonx_classifier_hyperparameters.yaml',
                                       model_endpoint=model_endpoint,
                                       project_id=project_id,
                                       )
    elif model_provider.lower() == 'openai':
        model_client = FoundationModel(model_provider=model_provider,
                                       api_key=api_key,
                                       config_path='configs/openai_classifier_hyperparameters.yaml')
    else:
        pass

    # get a particular instance of the model of choice
    model = model_client.instantiate_model()
    model_name = model_client.model_name

    return {'name': model_name,
            'model': model}


def get_prompt_template(file_name):
    """
    Get prompt template from file.
    :param file_name: the name of the file for the prompt template, including the .txt extension.
    :return: prompt str
    """
    file_handler.get_prompt_from_file(file_name)
    prompt_template = file_handler.prompt
    return prompt_template


def get_data_df(file_name):
    """
    Get data dataframe from file.
    :param file_name: the name of the file for the data to move to DataFrame.
    :return: pandas DataFrame
    """
    file_handler.get_data_from_file(file_name=file_name)
    df = file_handler.data
    return df


def prompt_inputs(topic, input_text):
    return {topic: input_text}


def run_article_classifier(model_provider):
    """
    Run the entire pipeline E2E.
    """
    # prepare the credentials
    credentials = prepare_credentials()

    # instantiate variables
    langchain_model = ''
    model_name = ''
    # get the model of choice
    if model_provider.lower() == 'watsonx':
        model_dict = get_model(
            model_provider=model_provider,
            api_key=credentials['watsonx_api_key'],
            model_endpoint=credentials['model_endpoint'],
            project_id=credentials['project_id']
        )
        base_model = model_dict['model']
        model_name = model_dict['name']

        # integrate with langchain Watsonx LLM model
        langchain_model = WatsonxLLM(model=base_model)

    elif model_provider.lower() == 'openai':
        model_dict = get_model(
            model_provider=model_provider,
            api_key=credentials['openai_api_key']
        )
        langchain_model = model_dict['model']
        model_name = model_dict['name']
    else:
        pass

    # get the prompt template
    red_flag_template = get_prompt_template(file_name='classify_article.txt')
    prompt_template = PromptTemplate.from_template(red_flag_template)

    # get the data
    df = get_data_df(file_name='First200_ic.csv')
    sample_articles = df[['_id',
                          'article',
                          'classification.isIncident']].loc[df['classification.isIncident'] == 'Incident']

    sample_articles = sample_articles.sample(10)

    # instantiate model
    llm_chain = LLMChain(prompt=prompt_template, llm=langchain_model)

    # new col name
    new_col = model_name + '_classification'
    # apply the model on the sample articles and store in a new column.
    sample_articles[new_col] = sample_articles['article'].apply(lambda x:
                                                                llm_chain.run(
                                                                    prompt_inputs('article', x)
                                                                )
                                                                )
    # standardize the output format.
    sample_articles.set_index('_id', inplace=True)

    # save the new output to data outputs.
    file_handler.save_df_to_file(df=sample_articles, file_name='sample_classification2.csv')
    print(sample_articles)
