from utils.files_handler import FileHandler
from utils.preprocess_text import StandardTextCleaner
from src.summarize_text import Summarizer
from src.text_classifier import TextClassifier
from src.sentiment_classifier import SentimentClassifier
from time import time

file_handler = FileHandler()
standard_cleaner = StandardTextCleaner()


def run_complaints_analysis(
        text_col='Consumer complaint narrative',
        primary_id='Complaint ID'
):
    # first get the data.
    data = file_handler.get_df_from_file('customer_complaints_sample.csv')

    print("Preprocessing the complaints text data")
    # preprocess the data.
    data[text_col] = data[text_col].apply(lambda x:
                                          standard_cleaner.remove_special_characters(text=x,
                                                                                     remove_markdown=True,
                                                                                     remove_special_chars=True,
                                                                                     remove_brackets=True,
                                                                                     remove_non_english=False
                                                                                     )
                                          )
    print("Preprocessing Complete")

    t1 = time()
    """Summarizer"""
    print(f"Summarizing {text_col}")
    summarizer = Summarizer()
    summary_col = summarizer.model_name + '_summary'
    # apply the model on text.
    data[summary_col] = data[text_col].apply(lambda x: summarizer.summarize_text(x))
    t2 = time() - t1
    print(f"Summarizer Complete in {t2}")

    """Categories Classifier"""
    print(f"Classifying {text_col} into Complaints Categories")
    category_classifier = TextClassifier(
        prompt_file_name='complaints_categories_classifier.txt',
        task='COMPLAINT_CATEGORY_CLASSIFIER'
    )
    new_col = category_classifier.model_name + '_category_classification'
    # apply the model on text.
    data[new_col] = data[summary_col].apply(lambda x: category_classifier.classify_text(
        topic='complaint',
        input_text=x)
                                         )
    t3 = time() - t2
    print(f"Category Classification Complete in {t3}")

    """Criteria Classifier"""
    print(f"Classifying {text_col} into Complaints Criteria")
    criteria_classifier = TextClassifier(
        prompt_file_name='complaints_criteria_classifier.txt',
        task='COMPLAINT_CRITERIA_CLASSIFIER'
    )
    new_col = criteria_classifier.model_name + '_criteria_classification'
    # apply the model on text.
    data[new_col] = data[summary_col].apply(lambda x: criteria_classifier.classify_text(
        topic='complaint',
        input_text=x)
                                         )
    t4 = time() - t3
    print(f"Criteria Classification Complete in {t4}")

    """Sentiment Classifier"""
    print(f"Classifying {text_col} into Sentiments")
    sentiment_classifier = SentimentClassifier()
    new_col = sentiment_classifier.model_name + '_sentiment_classification'
    # apply the model on text.
    data[new_col] = data[text_col].apply(lambda x: sentiment_classifier.classify_text_sentiment(
        topic='complaint',
        input_text=x)
                                         )
    t5 = time() - t4
    print(f"Sentiment Classification Complete in {t5}")

    """OUTPUT"""
    # standardize the output format.
    try:
        data.set_index(primary_id, inplace=True)
    except:
        pass

    for col_name in data.columns:
        if 'classification' in col_name:
            data[col_name] = data[col_name].apply(lambda x: standard_cleaner.remove_new_lines(x).capitalize())

    file_handler.save_df_to_csv(df=data, file_name='complaints_analysis.csv')
    print(f"Complete in {time() - t1}")
