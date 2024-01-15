import re
import string
import pandas as pd


def clean_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s@.]', '', text)

    # Remove non-English characters
    text = ''.join(char for char in text if char in string.ascii_letters or char.isspace() or char in "@.")

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    return text
