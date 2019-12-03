from sklearn.pipeline import Pipeline
from gensim.sklearn_api import lsimodel, ldamodel
from topic_analysis.nlp.normalization import TextNormalizer
from topic_analysis.nlp.vectorization import GensimTfidfVectorizer


class GensimTopicModel(object):

    def __init__(self, n_topics=25, estimator='LDA', stop_words_addition=[]):
        """
        n_topics is the desired number of topics
        To use Latent Semantic Analysis, set estimator to 'LSA'
        otherwise defaults to Latent Dirichlet Allocation.
        """
        self.n_topics = n_topics

        if estimator == 'LSA':
            self.estimator = lsimodel.LsiTransformer(num_topics=self.n_topics)
        else:
            self.estimator = ldamodel.LdaTransformer(num_topics=self.n_topics)

        self.model = Pipeline([
            ('normalization', TextNormalizer(stop_words_addition)),
            ('vectorization', GensimTfidfVectorizer()),
            ('model', self.estimator)
        ])

    def fit(self, documents):
        self.model.fit(documents)

        return self.model