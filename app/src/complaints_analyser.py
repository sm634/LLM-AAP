from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.files_handler import FileHandler
from utils.models_funcs import get_model
from src.summarize_text import Summarizer
from src.text_classifier import TextClassifier
from src.sentiment_classifier import SentimentClassifier

file_handler = FileHandler()


def run_complaints_analysis():

    # first get the data.


    summarizer = Summarizer()
    category_classifier = TextClassifier(prompt_file_name='complaints_categories_classifier.txt')
    criteria_classifier = TextClassifier(prompt_file_name='complaints_criteria_classifier.txt')
    sentiment_classifier = SentimentClassifier()

