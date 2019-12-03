from topic_analysis.nlp.topic_modeling import GensimTopicModel
import pyLDAvis
import pyLDAvis.gensim
from operator import itemgetter


def run_analysis_gensim(documents, n_topics, approach):

    # update with common words in Changes
    stop_words_addition = ['a', 'an', 'and', 'no',  'whom', 'from', 'or', 'for',
                          'like','get', 'let', 'want', 'like', 'do', 'dont', 'wont', 'would' 'with', 'to', 'how']

    # NOTE: Using TF-IDF we can forget about the stop words since automatically will be less relevant
    gensim_model = GensimTopicModel(n_topics, estimator = approach, stop_words_addition= stop_words_addition,)
    gensim_model.fit(documents)

    model = gensim_model.model.named_steps['model'].gensim_model

    corpus = [
        gensim_model.model.named_steps['vectorization'].lexicon.doc2bow(doc)
        for doc in gensim_model.model.named_steps['normalization'].transform(documents)
    ]


    lexicon = gensim_model.model.named_steps['vectorization'].lexicon

    # VISUALIZATION
    visualization_data = pyLDAvis.gensim.prepare(model, corpus, lexicon)
    pyLDAvis.save_html(visualization_data, 'gensim_lda.html')

    #PRINT TOPICS
    topics = get_topics(corpus, model)

    topic_document_list = sorted(list(zip(topics, documents)), key=itemgetter(0))

    for topic, doc in topic_document_list:
        print("Topic:{}".format(topic))
        print(doc)


    # Compute Perplexity
    print('\nPerplexity: ', model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.


def get_topics(vectorized_corpus, model):

        topics = [
            max(model[doc], key=itemgetter(1))[0]
            for doc in vectorized_corpus
        ]
        return topics

