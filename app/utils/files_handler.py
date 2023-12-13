import pandas as pd


class FileHandler:
    def __init__(self):

        # attributes for prompt files.
        self.prompt = ''
        self.prompts_folder_path = 'prompts/prompt_templates/'

        # attributes for data files.
        self.data = pd.DataFrame()
        self.data_input_folder_path = 'data/input/'
        self.data_output_folder_path = 'data/output/'

    def get_prompt_from_file(self, file_name):
        """
        Retrieves the content from a prompt template .txt file and stores it in a variable as type string.
        :param file_name: the name of the file with the prompt template.
        :param file_path: the file path to the prompt template of interest. e.g.
        'prompt_templates/red_flags_prompts1.txt'
        :return: the prompt template as string.
        """
        # ensure we read a .txt file to get the prompt template.
        assert '.txt' in file_name, ("The file is not of extension .txt. Please ensure it is, or include the extension"
                                     "in the argument.")

        file_path = self.prompts_folder_path + file_name
        # read the file.
        with open(file_path, 'r') as f:
            prompt = f.read()

        self.prompt = prompt

    def get_data_from_file(self, file_name):
        """
        Function to retrieve data as a pandas DataFrame from the designated data input folder.
        :param file_name: the name of the file (.csv) only currently.
        :return: a pandas DataFrame of the tabular data.
        """
        assert '.csv' in file_name, ("The file is not of extension .csv. Please ensure it is, or include the extension"
                                     "in the argument.")

        file_path = self.data_input_folder_path + file_name
        # read the file
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')

        self.data = df

    def save_df_to_file(self, df, file_name):
        """
        Function to retrieve data as a pandas DataFrame from the designated data input folder.
        :param df: the pandas DataFrame to save as a csv.
        :param file_name: the name of the csv file
        :return: a pandas DataFrame of the tabular data.
        """

        file_path = self.data_output_folder_path + file_name
        df.to_csv(file_path)
