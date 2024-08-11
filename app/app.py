import streamlit as st
import time

import pandas as pd
import time

from utils.files_handler import FileHandler
from connectors.vector_db_connector import MilvusConnector

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.models_funcs import get_model

## SET UP
file_handler = FileHandler()
file_handler.get_config()
config = file_handler.config
client = MilvusConnector()

def parse_uploaded_file():
    if uploaded_file is not None:
        # To read file as string
        file_content = uploaded_file.read().decode("utf-8")
        # Get the list of FAQs in a list.
        questions_list = file_content.split('\n')
        return questions_list

def prompt_inputs(topic1, passages, topic2, question):
    """
    Temporary function for the policy comparator which takes two arguments, which is mapped to the input data.
    :param topic1: The topic/name of the argument to be passed into the prompt template.
    :param passage1: The passage that is passed as the first comparator.
    :param topic2: The topic/name of the second argument to be passed into the prompt template.
    :param passage2: The 2nd passage that is passed as the 2nd comparator.
    :return: A dictionary that can be passed to a Langchain run command.
    """
    return {topic1: passages, topic2: question}


def faq_generator_setup(prompt_file):
    # get the model
    model_dict = get_model()
    model = model_dict['model']
    model_name = model_dict['name']

    # prepare the prompt template.
    prompt_file = file_handler.get_prompt_template(file_name=prompt_file)
    prompt_template = PromptTemplate.from_template(prompt_file)

    return{
        'model': model,
        'model_name': model_name,
        'prompt_template': prompt_template
    }

def get_faq_answer(model, prompt_template, topic1, passages, topic2, question):

    # instantiate model with prompt.
    llm_chain = LLMChain(prompt=prompt_template, llm=model)
    llm_response = llm_chain.invoke(
        prompt_inputs(topic1, passages, topic2, question)
    )

    return {
        'llm_response': llm_response['text']
    }

# store the metadata variables, which can be used to display certain answers.
collections_names = client.list_collections()

# Define configuration options
config_options = {
    "Collections": collections_names
}

# Streamlit UI components
st.title("WatsonX ModelRecommender")
st.subheader("Accelerate aligning WatsonX foundation models to Use Cases.")
st.write("      Nikhita Bagga, CSM (nikhita.bagga@ibm.com)")
st.write("      Alejandro Navarro, CE (alejandro.navarro@ibm.com)")
st.write("      Safal Mukhia, CSM (safal.mukhia@ibm.com)")

# create a subheading
st.subheader("Collections to Search")

collection = st.selectbox("Collection", config_options['Collections'])

st.subheader("Search with a question, or upload a file of queries")
col1, col2 = st.columns([1, 1])

# with col1:
query = st.text_input("Question or Query")
if st.button("Generate Model Recommendation") and query is not None:
    st.subheader("Retrieving Documents")

    # retrieve data.
    results_dict = client.search(
        collection_name=collection,
        query=query,
        fields=["text", "url"],
        anns_field='embeddings'
    )
    results = results_dict["text"]
    urls = results_dict["url"]

    input_passage = ""
    for i, text in enumerate(results):
        input_passage += f"Passage {i + 1}\n"
        input_passage += text + "\n"
        st.write(f"Document {i+1}")
        st.write(text)
        st.write(f"Source: {urls[i]}")
    
    # set up model and prompt
    faq_setup = faq_generator_setup(prompt_file='generate_answer.txt')
    model = faq_setup['model']
    model_name = faq_setup['model_name']
    prompt_template = faq_setup['prompt_template']

    model_response = get_faq_answer(
        model=model,
        prompt_template=prompt_template,
        topic1='passages',
        passages=input_passage,
        topic2='question',
        question=query
    )
    st.subheader(f"Model {model_name} Answer:")
    st.write(model_response["llm_response"])

else:
    st.write("Please insert a question.")

# process logic for uploaded files.
uploaded_file = st.file_uploader("Upload a file use case specifications to generate a report", type=["txt", "csv"])
if st.button("Generate Report for best Use Cases"):

    t1 = time.time()

    # set up for task with config
    config["TASK"] = "GENERATE_RECOMMENDATION_REPORT"
    file_handler.write_config(config)

    # set up model and prompt
    faq_setup = faq_generator_setup(prompt_file='generate_answer.txt')
    model = faq_setup['model']
    model_name = faq_setup['model_name']
    prompt_template = faq_setup['prompt_template']

    st.subheader("Generating FAQ answers into list")
    faq_document = parse_uploaded_file()
    st.write("Generating responses for the questions provided below:")
    st.write(faq_document)

    # prepare output fields.
    questions = []
    passages = []
    llm_answers = []

    # for progress bar
    progress_bar = st.progress(0)
    n_iterations = len(faq_document)

    for i, query in enumerate(faq_document):

        # update progress bar
        progress_percentage = (i + 1) / n_iterations
        progress_bar.progress(progress_percentage, f'Answering question {i + 1}')

        # retrieve data.
        results_dict = client.search(
            collection_name=collection,
            query=query,
            fields=["text"]
        )
        results = results_dict["text"]
        
        input_passage = ""
        for i, text in enumerate(results):
            input_passage += f"Passage {i + 1}\n"
            input_passage += text + "\n"

        model_response = get_faq_answer(
            model=model,
            prompt_template=prompt_template,
            topic1='passages',
            passages=input_passage,
            topic2='question',
            question=query
        )

        questions.append(query.replace("\n", "").replace(",", "").replace('\r', ''))
        passages.append(input_passage.replace("\n", "").replace(",", "").replace('\r', ''))
        llm_answers.append(model_response["llm_response"].replace("\n", "").replace(",", "").replace('\r', ''))

        time.sleep(0.1) 

    # prepare output dataframe
    generation_col_name = model_name + "_answer"
    df = pd.DataFrame(
        {
            "question": questions,
            "retrieved_document_passages": passages,
            generation_col_name: llm_answers
        }
    )

    output_file_name = f'faq_answers_{model_name}'
    file_handler.save_df_to_csv(df=df, file_name=output_file_name)

    st.subheader("Task Complete. Output produced.")

    t2 = time.time() - t1

    st.write(f"The task took {t2: .3f} seconds")