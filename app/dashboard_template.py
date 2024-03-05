import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


st.set_page_config(
    page_title="Complaints Handler Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Sample data
np.random.seed(0)
data = pd.DataFrame({
    'Date': pd.date_range(start='2022-01-01', end='2022-12-31'),
    'Value': np.random.randn(365)
})

# Sidebar
st.sidebar.header('Sidebar')
sidebar_option = st.sidebar.selectbox('Select Option', ['Option 1', 'Option 2', 'Option 3'])

# Create columns with different sizes
left_column, center_column, right_column = st.columns([1.5, 4.5, 2.5], gap='medium')

# Column 1: Cards with summary statistics
with left_column:
    st.subheader('Column 1: Summary Statistics')
    mean_value = np.mean(data['Value'])
    st.metric(label="Mean Value", value=mean_value)

    median_value = np.median(data['Value'])
    st.metric(label="Median Value", value=median_value)

    max_value = np.max(data['Value'])
    st.metric(label="Max Value", value=max_value)

    min_value = np.min(data['Value'])
    st.metric(label="Min Value", value=min_value)

# Column 2: Line Chart
with center_column:
    st.subheader('Column 2: Line Chart')
    line_chart_data = alt.Chart(data).mark_line().encode(
        x='Date',
        y='Value'
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(line_chart_data, use_container_width=True)

# Column 3: Table
with right_column:
    st.subheader('Column 3: Table')
    st.write(data)
