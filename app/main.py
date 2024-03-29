from src.text_classifier import TextClassifier
from src.article_redflag_comparator import run_article_redflag_comparator
from src.preprocess_pipeline import run_preprocess_pipeline
from src.text_comparator import run_text_comparator
from src.embeddings_comparator import run_embeddings_comparator
from src.extract_fields import extract_fields
from src.summarize_text import Summarizer
from src.complaints_analyser import run_complaints_analysis

from utils.files_handler import FileHandler

file_handler = FileHandler()


def main():
    # get models config
    file_handler.get_config()
    config = file_handler.config
    # get arguments config
    file_handler.get_config('arguments_passer.yaml')
    arguments_config = file_handler.config

    task = config['TASK'].lower()
    print("Running task: ", task)
    if task == 'text_classifier':
        text_classifier = TextClassifier('customer_complaints_sample.csv')
        text_classifier.run_text_classifier()
    elif task == 'redflag_article_comparator':
        run_article_redflag_comparator()
    elif task == 'preprocess_article':
        run_preprocess_pipeline(use_standard_cleaner=False, use_denoiser=True)
    elif task == 'text_comparator':
        llm_analysis = arguments_config['TEXT_COMPARATOR']['INVOKE_LLM_ANALYSIS']
        llm_generation = arguments_config['TEXT_COMPARATOR']['INVOKE_LLM_GENERATION']
        run_text_comparator(invoke_llm_analysis=llm_analysis,
                            invoke_llm_generation=llm_generation)
    elif task == 'embeddings_comparator':
        llm_analysis = arguments_config['EMBEDDINGS_COMPARATOR']['INVOKE_LLM_ANALYSIS']
        llm_generation = arguments_config['EMBEDDINGS_COMPARATOR']['INVOKE_LLM_GENERATION']
        run_embeddings_comparator(invoke_llm_analysis=llm_analysis,
                                  invoke_llm_generation=llm_generation)
    elif task == 'extract_fields':
        fields = arguments_config['EXTRACT_FIELDS']['fields']
        extract_fields(fields)
    elif task == 'summarizer':
        summarizer = Summarizer()
        summarizer.run_summarizer(save_output=True)
    elif task == 'complaints_analyser':
        run_complaints_analysis()

    print("Task Complete")


if __name__ == '__main__':
    main()
