#import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.utils import simple_preprocess
import nltk

class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    1. TOKENIZATION
    2. LEMMATIZATION
    """

    def __init__(self, extended_stopwords_list=None, language="english"):

        # stop words configuration: language and extension
        self.download_stopwords()
        self.stopwords = nltk.corpus.stopwords.words(language)
        self.stopwords.extend(extended_stopwords_list)

        # Initialize lemmatizer
        self.lemmatizer = nltk.WordNetLemmatizer()


    def download_stopwords(self):
        try:
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('stopwords')


    def normalize(self, document):

        """
        gensim.simple_process()
        - deacc=True removes punctuations
        - min_len = dont process word with less than minimum length
        """
        return [self.lemmatize(word)
                for word in simple_preprocess(document, deacc=True, min_len=3)
                if word not in self.stopwords]

    def lemmatize(self, token):

        return self.lemmatizer.lemmatize(token, pos='v')  # assume is a verb: broader but this can be improved

    def fit(self, X=None, y=None):
        return self

    def transform(self, documents):
        normalized_documents = list()
        for document in documents:
            normalized_doc = self.normalize(document)
            if len(normalized_doc) > 0:
                normalized_documents.append(normalized_doc)
        return normalized_documents