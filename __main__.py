import pandas as pd

from topic_analysis import topic_analysis_pipeline

def load_data(path):
    return pd.read_csv(path, delimiter=',')

dataset = load_data("./data/job_skills_final.csv")


EXTRACTION_STRATEGY = 'first_message' # 'first_message'

N_TOPICS = 12

MODELING_APPROACH = 'LDA' #LSA


if __name__ == '__main__':

    #documents_list = slack_channel_reader.get(fromFile=False, document_strategy=EXTRACTION_STRATEGY)

    topic_analysis_pipeline.run_analysis_gensim(list(dataset), n_topics=N_TOPICS, approach=MODELING_APPROACH)


