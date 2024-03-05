from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.files_handler import FileHandler
from utils.models_funcs import get_model
from src.summarize_text import Summarizer
from src.text_classifier import TextClassifier

file_handler = FileHandler()


