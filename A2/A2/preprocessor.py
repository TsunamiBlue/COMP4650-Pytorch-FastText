import nltk
from functools import lru_cache


class Preprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stermmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.stem = lru_cache(maxsize=100000)(nltk.SnowballStemmer(language='english').stem)
        # self.stem = lru_cache(maxsize=100000)(nltk.PorterStemmer().stem)
        self.tokenize = nltk.tokenize.WordPunctTokenizer().tokenize
        # self.tokenize = nltk.tokenize.WhitespaceTokenizer().tokenize

    def __call__(self, text):
        tokens = self.tokenize(text)
        tokens = [self.stem(token) for token in tokens]
        return tokens
