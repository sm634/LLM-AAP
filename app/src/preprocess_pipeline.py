"""
Docstring
---------
Current code assumes the 'trafficking' article. Needs updating. This needs to be updated and generalised.
"""

from utils.preprocess_text import TextDenoiser, StandardTextCleaner
from utils.models_funcs import get_model
from utils.files_handler import FileHandler

file_handler = FileHandler()


def run_preprocess_pipeline(file_name='test_sampled_set.csv',
                            use_standard_cleaner=False,
                            use_denoiser=True):
    """
    Run the article/text cleaning preprocess pipeline.
    :param file_name: The name of the file containing articles/text that needs to be preprocessed.
    :param use_standard_cleaner: Bool, option to use standard cleaning function on the text field.
    :param use_denoiser: Bool, option to use noise extraction with LLM on the text field.
    :return: pandas Dataframe with cleaned articles column.
    """
    # initialize denoiser model (selected from config)
    model_dict = get_model()
    model = model_dict['model']
    model_name = model_dict['name']
    print(f"Instantiated Denoiser model: {model_name}")

    df = file_handler.get_df_from_file(file_name=file_name)
    sample_articles = df[['_id',
                          'article',
                          'classification.isIncident']].loc[df['classification.isIncident'] == 'Incident']

    # apply standard cleaner on articles.
    if use_standard_cleaner:
        # instantiate standard cleaner as first step of preprocessing.
        standard_cleaner = StandardTextCleaner()
        standard_col = 'standard_cleaned_articles'
        sample_articles[standard_col] = sample_articles['article'].apply(
            lambda x: standard_cleaner.remove_special_characters(x)
        )
        print("Standard cleaning complete.")
    else:
        standard_col = 'article'

    # apply denoiser if applicable.
    if use_denoiser:
        # instantiate denoiser using LLMs
        denoiser = TextDenoiser()
        noise_removed_col = f'cleaned_article_{model_name}'

        sample_articles[noise_removed_col] = sample_articles[standard_col].apply(
            lambda x: denoiser.remove_adverts(input_text=x,
                                              prompt_template_file='remove_ads.txt',
                                              llm_model=model)
        )
        print("Denoiser model preprocessing complete.")

    # output name.
    output_file_name = f'cleaned_articles.csv'
    file_handler.save_df_to_csv(
        df=sample_articles,
        file_name=output_file_name
    )
    print("Output saved to ", file_handler.data_output_folder_path + output_file_name)

    return sample_articles
