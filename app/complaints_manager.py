import streamlit as st
import numpy as np
import altair as alt
from utils.files_handler import FileHandler
from utils.preprocess_text import StandardTextCleaner

file_handler = FileHandler()
standard_cleaner = StandardTextCleaner()

st.set_page_config(
    page_title="Complaints Handler Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Sample data
np.random.seed(0)
data = file_handler.get_df_from_file("complaints_analysis_2024-03-06-13-39-41-676908.csv")
data['LLAMA_2_70B_CHAT_criteria_classification'] = data['LLAMA_2_70B_CHAT_criteria_classification'].apply(
    lambda x: standard_cleaner.remove_new_lines(x).capitalize()
)
data['GRANITE_13B_CHAT_V2_category_classification'] = data['GRANITE_13B_CHAT_V2_category_classification'].apply(
    lambda x: standard_cleaner.remove_new_lines(x).capitalize()
)
data['GRANITE_13B_CHAT_V2_sentiment_classification'] = data['GRANITE_13B_CHAT_V2_sentiment_classification'].apply(
    lambda x: standard_cleaner.remove_new_lines(x).capitalize()
)
# Sidebar
st.sidebar.header('Filter by Data')
sidebar_option = st.sidebar.selectbox('Select Option to view data by',
                                      ['Complaint Categories', 'Complaint Criteria', 'Complaint Sentiment'])
sidebar_button = st.sidebar.button('Sample Summary')

# Create columns with different sizes
left_column, right_column = st.columns([3, 7], gap='medium')

# Column 1: Cards with summary statistics
with left_column:
    st.subheader('Summary Statistics')
    total_complaints = data.shape[0]
    st.metric(label="Total Complaints", value=total_complaints)

    # median_value = np.median(data['Value'])
    # st.metric(label="Median Value", value=median_value)
    #
    # max_value = np.max(data['Value'])
    # st.metric(label="Max Value", value=max_value)
    #
    # min_value = np.min(data['Value'])
    # st.metric(label="Min Value", value=min_value)

if sidebar_option == 'Complaint Categories':
    categories_by_date = data.groupby(['Date received', 'GRANITE_13B_CHAT_V2_category_classification']).count()
    categories_by_date = categories_by_date[categories_by_date.columns[0]]
    categories_by_date = categories_by_date.reset_index()
    with right_column:
        st.subheader('Complaints Categories by Date')
        st.line_chart(categories_by_date,
                      x='Date received',
                      y='Consumer complaint narrative',
                      color='GRANITE_13B_CHAT_V2_category_classification')

elif sidebar_option == 'Complaint Criteria':
    criteria_by_date = data.groupby(['Date received', 'LLAMA_2_70B_CHAT_criteria_classification']).count()
    criteria_by_date = criteria_by_date[criteria_by_date.columns[0]]
    criteria_by_date = criteria_by_date.reset_index()
    with right_column:
        st.subheader('Complaints Criteria by Date')
        st.line_chart(criteria_by_date,
                      x='Date received',
                      y='Consumer complaint narrative',
                      color='LLAMA_2_70B_CHAT_criteria_classification')

elif sidebar_option == 'Complaint Sentiment':
    sentiments_by_date = data.groupby(['Date received', 'GRANITE_13B_CHAT_V2_sentiment_classification']).count()
    sentiments_by_date = sentiments_by_date[sentiments_by_date.columns[0]]
    sentiments_by_date = sentiments_by_date.reset_index()
    with right_column:
        st.subheader('Complaints Sentiments by Date')
        st.line_chart(sentiments_by_date,
                      x='Date received',
                      y='Consumer complaint narrative',
                      color='GRANITE_13B_CHAT_V2_sentiment_classification')

# Summaries Table
st.subheader('Summaries')
st.write(data[['Consumer complaint narrative', 'GRANITE_13B_CHAT_V2_summary']].sample(1))
if sidebar_button:
    st.write(data[['Consumer complaint narrative', 'GRANITE_13B_CHAT_V2_summary']].sample(1))
