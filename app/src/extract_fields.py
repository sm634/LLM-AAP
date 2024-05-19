from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.models_funcs import get_model
from utils.files_handler import FileHandler
from src.preprocess_pipeline import run_preprocess_pipeline


file_handler = FileHandler()


def prompt_inputs(key1, fields, key2, text):
    """
    Temporary function for article classifier which takes one argument, which is to be mapped ot the input data.
    :param topic: The topic/name of the argument to be passed into the prompt template.
    :param input_text: The input/text that is passed as an article.
    :return: A dictionary that can be passed to a Langchain run command.
    """
    return {key1: fields, key2: text}


def extract_fields(fields: str,
                   file_name='cleaned_articles_2024-02-29-10-09-59-630916.csv',
                   preprocess=False,
                   standard_col='article'):
    """
    A function that extracts a set of fields from the provided input_text.
    :param file_name: The name of the file to read the data from.
    :param fields: Str, A list of fields wrapped around quotation marks to extract from the input_text.
    :param input_text: The input text to perform the extraction on.
    :param preprocess: Bool, option to preprocess the text first.
    :param standard_col: The name of the standard text column in the dataframe that contains the text that the
    extraction will be performed on.
    :return: Pandas DataFrame containing a column with the extracted fields.
    """

    df = file_handler.get_df_from_file(file_name=file_name)
    sample_articles = df[['_id',
                          standard_col]]

    if preprocess:
        print("Running Preprocessing Pipeline")
        sample_articles = run_preprocess_pipeline(file_name=file_name)

    # initialize the field extracting model.
    model_dict = get_model()
    model = model_dict['model']
    model_name = model_dict['name']
    print(f"Instantiated Extraction Model: {model_name}")

    # set up the prompt template
    field_extraction_template = file_handler.get_prompt_template(file_name='fields_extraction.txt')
    prompt_template = PromptTemplate.from_template(field_extraction_template)
    # set up the variables.
    fields_to_extract = f"{fields}"

    llm_chain = LLMChain(prompt=prompt_template, llm=model)

    fields_extracted_col = f'extracted_fields_{model_name}'

    sample_articles[fields_extracted_col] = sample_articles[standard_col].apply(
        lambda x: llm_chain.invoke(
            prompt_inputs('fields', fields_to_extract, 'text', x)
        )['text']
    )
    print("Field Extraction Complete.")

    output_file_name = f'extracted_fields.csv'
    file_handler.save_df_to_csv(
        df=sample_articles,
        file_name=output_file_name
    )

    print("Sample Article Extraction Complete.")
    return sample_articles
