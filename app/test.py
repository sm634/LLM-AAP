import pandas as pd

summary_classification = pd.read_csv(
    "C:/Users/SafalMukhia/projects/LLM-AAP/app/data/output/complaints_analysis_classification_summary.csv"
)
raw_classification = pd.read_csv(
    "C:/Users/SafalMukhia/projects/LLM-AAP/app/data/output/complaints_analysis_classification_raw.csv"
)

cols = ['Unnamed: 0',
        'Date received',
        'GRANITE_13B_CHAT_V2_category_classification',
        'LLAMA_2_70B_CHAT_criteria_classification',
        'GRANITE_13B_CHAT_V2_sentiment_classification']

summary_classification = summary_classification[cols]
raw_classification = raw_classification[cols]

df_concat = pd.concat([raw_classification, summary_classification], axis=0).drop_duplicates(inplace=False)

breakpoint()

df_concat['Complaint Text'] = raw_classification['Consumer complaint narrative']
df_concat['Summary_for_raw_classification'] = raw_classification['GRANITE_13B_CHAT_V2_summary']
df_concat['Summary_for_summary_classification'] = summary_classification['GRANITE_13B_CHAT_V2_summary']

df_concat.to_csv("data/output/summary_vs_raw_classification_output.csv")
