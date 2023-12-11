import pandas as pd


def local_file_df(file_path: str):
    """
    Function to read a file from local directory.
    :param file_path: file path, can be relative or absolute.
    :return: pandas DataFrame of the file read.
    """
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1')

    return df
