from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.files_handler import FileHandler
from utils.models_funcs import get_model

# global variable to process required files for data and prompts.
file_handler = FileHandler()


class TextClassifier:

    def __init__(self, prompt_file_name, task):
        # initialize model
        model_dict = get_model(task=task)
        self.model = model_dict['model']
        self.model_name = model_dict['name']

        # prompts init
        prompt_template = file_handler.get_prompt_template(file_name=prompt_file_name)
        self.classifier_template = PromptTemplate.from_template(prompt_template)

        # instantiate model
        self.llm_chain = LLMChain(prompt=self.classifier_template, llm=self.model)

    @staticmethod
    def prompt_inputs(topic, input_text):
        """
        Temporary function for article classifier which takes one argument, which is to be mapped ot the input data.
        :param topic: The topic/name of the argument to be passed into the prompt template.
        :param input_text: The input/text that is passed as an article.
        :return: A dictionary that can be passed to a Langchain run command.
        """
        return {topic: input_text}

    def classify_text(self, topic, input_text: str) -> str:
        """
        A function used to invoke the text summarizer on a single piece of text.
        """
        classification = self.llm_chain.invoke(
            self.prompt_inputs(topic, input_text)
        )
        return classification['text']

    def run_text_classifier(self,
                            input_file='customer_complaints_sample.csv',
                            text_col='Consumer complaint narrative'):
        """
        Run the entire pipeline E2E.
        """

        # get the data
        df = file_handler.get_df_from_file(file_name=input_file)

        # new col name
        new_col = self.model_name + '_classification'
        # apply the model on the sample articles and store in a new column.
        df[new_col] = df[text_col].apply(lambda x: self.classify_text(text_col, x))

        """OUTPUT"""
        # standardize the output format.
        df.set_index('_id', inplace=True)

        # define the output name.
        output_name = f'sample_classification_{self.model_name}.csv'

        # save the new output to data outputs.
        file_handler.save_df_to_csv(df=df, file_name=output_name)

        return df
