import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor:
    def __init__(self, corpus):
        self.corpus = corpus
        self.stopwords = stopwords.words('arabic')

    def get_document_content(self, filename):
        """
        filename(str): file_name that we want to extract content from it.
        return(str): document content.
        """
        return self.corpus.raw(filename)
    
    def extract_metadata(self, doc):
        """
        Extract title, publisher, and content from `doc`.
        doc(str): document that we want to extract content from it.
        return(tuple): consisting of title, publisher, and content of document.
        """
        raw_content_list = re.findall(r'.+', doc)
        title = extract_title(raw_content_list)
        publisher = extract_publisher(raw_content_list)
        content = extract_body(raw_content_list)

        return title, publisher, content

    def extract_keywords(self, documents, ngram_range=(1, 1)):
        """
        documents(list): list of strings containing documents we want to extract key words.
        return(df): contains each file in a row, with all vocab words in columns.
        """
        tf_idf = TfidfVectorizer(ngram_range=ngram_range)
        X = tf_idf.fit_transform(documents).toarray()
        df = pd.DataFrame(X, columns=tf_idf.get_feature_names_out())
        df.index = [self.corpus.fileids()[i] for i in range(len(documents))]
        return df

    def clean_document(self, doc):
        """
        Performing preprocessing on `doc` including: remove stopwords,
        remove numbers, remove punctuation.
        doc(str): `doc` content that we want to clean.
        return(str): `cleaned_text` preprocessing result.
        """
        tokens = doc.split()
        non_stopwords = [token for token in tokens if token not in self.stopwords]
        alpha_tokens = [token for token in non_stopwords if token.isalpha()]
        cleaned_text = ' '.join(alpha_tokens)

        return cleaned_text

    
def extract_title(content_list):
    """Extract title from document content.
    content_list(list): contains content of a document separated by new lines.
    returns(str): title of a document.
    """
    title = content_list[1]

    if '#' in title:
        title_list = list(title)
        title_list.remove('#')
        title = ''.join(title_list)

    return title.strip()


def extract_publisher(content_list):
    """Extract publisher from document content.
    content_list(list): contains content of a document separated by new lines.
    returns(str): publisher of a document.
    """
    publisher = content_list[3]
    return publisher.strip()


def extract_body(content_list):
    """Extract body from document content.
    content_list(list): contains content of a document separated by new lines.
    returns(str): body of a document.
    """
    body = content_list[5]
    return body.strip()
