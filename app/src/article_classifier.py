from datetime import datetime
from connectors.models_connector import ModelConnector
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.files_handler import FileHandler
from utils.timestamps import get_stamp

# global variables used in local functions
file_handler = FileHandler()


def get_model():
    """
    A function to instantiate the particular instance of the model desired.
    :return: Foundation model of choice.
    """

    model_client = ModelConnector()
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


def run_article_classifier():
    """
    Run the entire pipeline E2E.
    """
    # get the model
    model_dict = get_model()
    model = model_dict['model']
    model_name = model_dict['name']

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
    llm_chain = LLMChain(prompt=prompt_template, llm=model)

    # new col name
    new_col = model_name + '_classification'
    # apply the model on the sample articles and store in a new column.
    sample_articles[new_col] = sample_articles['article'].apply(lambda x:
                                                                llm_chain.run(
                                                                    prompt_inputs('article', x)
                                                                )
                                                                )

    """OUTPUT"""
    # standardize the output format.
    sample_articles.set_index('_id', inplace=True)
    print(sample_articles)

    # format to save file.
    stamp = get_stamp()
    output_name = f'sample_classification_{model_name}_{stamp}.csv'

    # save the new output to data outputs.
    file_handler.save_df_to_file(df=sample_articles, file_name=output_name)
