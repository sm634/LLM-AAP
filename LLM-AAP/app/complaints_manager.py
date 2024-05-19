import streamlit as st
import altair as alt

from utils.files_handler import FileHandler
from utils.preprocess_text import StandardTextCleaner
from charts.charts_func import donut_chart, line_chart

st.set_option('deprecation.showPyplotGlobalUse', False)

file_handler = FileHandler()
standard_cleaner = StandardTextCleaner()

st.set_page_config(
    page_title="Complaints Handler",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Load and format the Dataframe to be presentable.
data = file_handler.get_df_from_file("complaints_analysis_output.csv")
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
data['Complaint ID'] = data['Complaint ID'].astype(int)


def display_row(analytics_col):
    """
    A function that allows user query to surface a particular complaint entered at the bottom
    of the analytics page.
    :param analytics_col: The column to be displayed in the third row along with the actual complaint and summary.
    This will depend on the view that has been selected.
    """
    # Input complaint ID to filter by.
    complaint_id_input = st.text_input("Search by Complaint ID:", "", max_chars=5)
    # Create columns with different sizes
    left, middle, right = st.columns([1, 1, 1], gap='medium')
    # Adjust the width parameter to change the size of the input box
    if complaint_id_input:
        # Filter by complaint ID
        try:
            complaint_id = int(complaint_id_input)
            filtered_df = data[data['Complaint ID'] == complaint_id]
            with left:
                st.subheader("Complaint Text")
                st.write(filtered_df['Complaint Text'].iloc[0])
            with middle:
                st.subheader("Complaint Summary")
                st.write(filtered_df['GRANITE_13B_CHAT_V2_summary'].iloc[0])
            with right:
                st.subheader(f"Complaint {analytics_col}")
                st.write(filtered_df[analytics_col].iloc[0])

        except ValueError:
            st.error("Please enter a valid Complaint ID.")


def go_to_analytics_page():
    # Analytics page sidebar options.
    st.sidebar.header('Select Attribute to View Data')
    sidebar_option = st.sidebar.selectbox('Select Option to view data by',
                                          ['Complaint Categories', 'Complaint Criteria', 'Complaint Sentiment'])

    # Create columns with different sizes
    left_column, right_column = st.columns([5, 8], gap='medium')
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
            categories_by_date = data.groupby(['Month', 'Category']).count()
            count_col = categories_by_date.columns[0]
            categories_by_date = categories_by_date[count_col]
            categories_by_date = categories_by_date.reset_index()
            categories_by_date = categories_by_date.rename(columns={count_col: 'Count'})

            st.subheader('Complaints Categories by Date')
            line_chart(categories_by_date, chart_title='Complaints Categories by Date',
                       group_col='Category')
            st.pyplot()

        display_row('Category')

    elif sidebar_option == 'Complaint Criteria':
        criteria_by_date = data.groupby(['Month', 'Criteria']).count()
        count_col = criteria_by_date.columns[0]
        criteria_by_date = criteria_by_date[count_col]
        criteria_by_date = criteria_by_date.reset_index()
        criteria_by_date = criteria_by_date.rename(columns={count_col: 'Count'})

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
                          x='Month',
                          y='Count',
                          color='Criteria')

        display_row('Criteria')

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
            sentiments_by_date = data.groupby(['Month', 'Sentiment']).count()
            count_col = sentiments_by_date.columns[0]
            sentiments_by_date = sentiments_by_date[count_col]
            sentiments_by_date = sentiments_by_date.reset_index()
            sentiments_by_date = sentiments_by_date.rename(columns={count_col: 'Count'})

            st.subheader('Complaints Sentiments by Date')
            st.line_chart(sentiments_by_date,
                          x='Month',
                          y='Count',
                          color='Sentiment')

        display_row('Sentiment')


def go_to_playground_page():
    st.title("Complaints Recommendation")
    # st.sidebar.header('Select Attribute to View Data')
    # sidebar_option = st.sidebar.selectbox('Select Option to view data by',
    #                                       ['Summary', 'Category', 'Criteria'])

    # recommend_button = st.button("Analyse and make recommendation")
    # ### IMPLEMENT RECOMMENDATION SETTING HERE.


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ('Page 1 - Complaints Analytics', 'Page 2 - Complaints Playground'))

    if page == 'Page 1 - Complaints Analytics':
        go_to_analytics_page()
    elif page == 'Page 2 - Complaints Playground':
        go_to_playground_page()


if __name__ == "__main__":
    main()
