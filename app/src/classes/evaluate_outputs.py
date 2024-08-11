import Levenshtein
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
import pandas as pd

class EvaluateLLMOutputs:
    def __init__(self) -> None:
        pass

    @staticmethod
    # Function to compute normalized Levenshtein Distance
    def normalized_levenshtein_distance(str1, str2):
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 0
        return Levenshtein.distance(str1, str2) / max_len

    @staticmethod
    # Function to evaluate classification.
    def evaluate_classifier(df, y_test_col, y_pred_col):
        # Since this is going to be a text output, we will have to recode them into numbers first.
        # we will simply do iterative lable recoding.
        
        labels_encoded, unique_labels = pd.factorize(df[y_test_col])
        predictions_encoded, unique_labels = pd.factorize(df[y_pred_col])

        df[y_test_col] = labels_encoded
        df[y_pred_col] = predictions_encoded
        
        accuracy = accuracy_score(y_true=df[y_test_col], y_pred=df[y_pred_col])
        precision = precision_score(y_true=df[y_test_col], y_pred=df[y_pred_col], average='macro')
        recall = recall_score(y_true=df[y_test_col], y_pred=df[y_pred_col], average='macro')
        f1 = f1_score(y_true=df[y_test_col], y_pred=df[y_pred_col], average='macro')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
