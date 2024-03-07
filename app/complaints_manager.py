import streamlit as st
import altair as alt
import numpy as np

from utils.files_handler import FileHandler
from utils.preprocess_text import StandardTextCleaner
from charts.charts_func import donut_chart

st.set_option('deprecation.showPyplotGlobalUse', False)

file_handler = FileHandler()
standard_cleaner = StandardTextCleaner()

st.set_page_config(
    page_title="Complaints Handler Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Load and format the Dataframe to be presentable.
data = file_handler.get_df_from_file("complaints_analysis_2024-03-06-13-39-41-676908.csv")
data = data.rename(columns={'Date received': 'Complaint Date',
                            'LLAMA_2_70B_CHAT_criteria_classification': 'Criteria',
                            'GRANITE_13B_CHAT_V2_category_classification': 'Category',
                            'GRANITE_13B_CHAT_V2_sentiment_classification': 'Sentiment',
                            'Consumer complaint narrative': 'Complaint Text'})

data['Criteria'] = data['Criteria'].apply(
    lambda x: standard_cleaner.remove_new_lines(x).capitalize()
)
data['Category'] = data['Category'].apply(
    lambda x: standard_cleaner.remove_new_lines(x).capitalize()
)
data['Sentiment'] = data['Sentiment'].apply(
    lambda x: standard_cleaner.remove_new_lines(x).capitalize()
)
# Sidebar
st.sidebar.header('Select Attribute to View Data')
sidebar_option = st.sidebar.selectbox('Select Option to view data by',
                                      ['Complaint Categories', 'Complaint Criteria', 'Complaint Sentiment'])

# Create columns with different sizes
left_column, right_column = st.columns([4, 7], gap='medium')

if sidebar_option == 'Complaint Categories':

    with left_column:
        st.subheader('Summary Statistics')

        # criteria_breakdown is your DataFrame
        category_breakdown = data[
            ['Complaint Text',
             'Category']
        ].groupby('Category').count()
        category_breakdown = category_breakdown.reset_index()
        category_breakdown = category_breakdown.rename(columns={'Complaint Text': 'Count'})

        total_complaints = data.shape[0]
        st.metric(label="Total Complaints", value=total_complaints)

        # Display the chart on Streamlit
        donut_chart(data=category_breakdown, chart_title='Category Breakdown', legend_col='Category')
        st.pyplot()

    with right_column:
        categories_by_date = data.groupby(['Complaint Date', 'Category']).count()
        categories_by_date = categories_by_date[categories_by_date.columns[0]]
        categories_by_date = categories_by_date.reset_index()
        categories_by_date = categories_by_date.rename(columns={'Complaint Text': 'Count'})

        st.subheader('Complaints Categories by Date')
        st.line_chart(categories_by_date,
                      x='Complaint Date',
                      y='Count',
                      color='Category')

    # Summaries Table
    st.subheader('Sample Summaries and Categories')
    st.dataframe(data[['Complaint Text',
                       'GRANITE_13B_CHAT_V2_summary',
                       'Category']].sample(10))

elif sidebar_option == 'Complaint Criteria':
    criteria_by_date = data.groupby(['Complaint Date', 'Criteria']).count()
    criteria_by_date = criteria_by_date[criteria_by_date.columns[0]]
    criteria_by_date = criteria_by_date.reset_index()
    criteria_by_date = criteria_by_date.rename(columns={'Complaint Text': 'Count'})

    with left_column:
        st.subheader('Summary Statistics')
        total_complaints = data.shape[0]
        st.metric(label="Total Complaints", value=total_complaints)

        # sentiments_breakdown is your DataFrame
        criteria_breakdown = data[
            ['Complaint Text',
             'Criteria']
        ].groupby('Criteria').count()
        criteria_breakdown = criteria_breakdown.reset_index()
        criteria_breakdown = criteria_breakdown.rename(columns={'Complaint Text': 'Count'})

        # Display the chart on Streamlit
        donut_chart(data=criteria_breakdown, chart_title='Criteria Breakdown', legend_col='Criteria')
        st.pyplot()

    with right_column:
        st.subheader('Complaints Criteria by Date')
        st.line_chart(criteria_by_date,
                      x='Complaint Date',
                      y='Count',
                      color='Criteria')

    # Summaries Table
    st.subheader('Sample Summaries and Criteria')
    st.dataframe(data[['Complaint Text',
                       'GRANITE_13B_CHAT_V2_summary',
                       'Criteria']].sample(10))

elif sidebar_option == 'Complaint Sentiment':

    with left_column:
        st.subheader('Summary Statistics')
        total_complaints = data.shape[0]
        st.metric(label="Total Complaints", value=total_complaints)

        # sentiments_breakdown is your DataFrame
        sentiments_breakdown = data[
            ['Complaint Text',
             'Sentiment']
        ].groupby('Sentiment').count()
        sentiments_breakdown = sentiments_breakdown.reset_index()
        sentiments_breakdown = sentiments_breakdown.rename(columns={'Complaint Text': 'Count'})

        # Display the chart on Streamlit
        donut_chart(data=sentiments_breakdown, chart_title='Sentiment Breakdown', legend_col='Sentiment')
        st.pyplot()

    with right_column:
        sentiments_by_date = data.groupby(['Complaint Date', 'Sentiment']).count()
        sentiments_by_date = sentiments_by_date[sentiments_by_date.columns[0]]
        sentiments_by_date = sentiments_by_date.reset_index()
        sentiments_by_date = sentiments_by_date.rename(columns={'Complaint Text': 'Count'})

        st.subheader('Complaints Sentiments by Date')
        st.line_chart(sentiments_by_date,
                      x='Complaint Date',
                      y='Count',
                      color='Sentiment')

    # Summaries Table
    st.subheader('Sample Summaries and Criteria')
    st.dataframe(data[['Complaint Text',
                       'GRANITE_13B_CHAT_V2_summary',
                       'Sentiment']].sample(10))
