from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.files_handler import FileHandler
from utils.models_funcs import get_model

file_handler = FileHandler()


class Summarizer:

    def __init__(self, prompt_file_name='complaints_summarizer_prompt.txt'):
        # initialize model
        model_dict = get_model()
        self.model = model_dict['model']
        self.model_name = model_dict['name']

        # prompts init
        prompt_template = file_handler.get_prompt_template(file_name=prompt_file_name)
        self.summarizer_template = PromptTemplate.from_template(prompt_template)

        # instantiate model
        self.llm_chain = LLMChain(prompt=self.summarizer_template, llm=self.model)

    @staticmethod
    def prompt_inputs(topic, input_text):
        """
        Temporary function for article classifier which takes one argument, which is to be mapped ot the input data.
        :param topic: The topic/name of the argument to be passed into the prompt template.
        :param input_text: The input/text that is passed as an article.
        :return: A dictionary that can be passed to a Langchain run command.
        """
        return {topic: input_text}

    def summarize_text(self, input_text: str) -> str:
        """
        A function used to invoke the text summarizer on a single piece of text.
        """
        summary = self.llm_chain.invoke(
            self.prompt_inputs('complaint', input_text)
        )
        return summary['text']

    def run_summarizer(
            self,
            input_file_name='customer_complaints_sample.csv',
            prompt_file_name='complaints_summarizer_prompt.txt',
            complaints_col='Consumer complaint narrative',
            primary_id='Complaint ID',
            save_output=False
    ):
        """
        Run the entire pipeline E2E.
        """

        # get the data
        df = file_handler.get_df_from_file(file_name=input_file_name)
        sample_df = df.head(10)

        # new col name
        new_col = self.model_name + '_summary'
        # apply the model on the sample articles and store in a new column.
        sample_df[new_col] = sample_df[complaints_col].apply(lambda x:
                                                             self.llm_chain.invoke(
                                                                 self.prompt_inputs('complaint', x)
                                                                )['text']
                                                             )

        """OUTPUT"""
        # standardize the output format.
        sample_df.set_index(primary_id, inplace=True)

        if save_output:
            # define the output name.
            output_name = f'complaints_summary_{self.model_name}.csv'
            # save the new output to data outputs.
            file_handler.save_df_to_csv(df=sample_df, file_name=output_name)

        return sample_df
