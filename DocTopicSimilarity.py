import nltk
import pandas as pd
import string
import numpy as np
import math
import os

from collections import Counter
from nltk.tokenize import TreebankWordTokenizer
nltk.download('stopwords')



def read_docs(docs_file):
    docs = list()
    i = 0

    with open(docs_file, 'r') as docs_file:
        for line in docs_file:
            if line.startswith("#") and not i:
                string = ''
                i += 1
            elif line.startswith("#") and i:
                docs.append(string)
                i += 1
                string = ''
            else:
                string += line
        docs.append(string)

    assert len(docs) != 0
    return docs

def tokenize_doc(doc):
    tokenizer = TreebankWordTokenizer()
    tokens = list()

    for token in doc:
        tokens.append(tokenizer.tokenize(token))

    return tokens

def clean_tokens(tokens):
    stop_words = nltk.corpus.stopwords.words('english')
    clean_tokens = list()
    clean_no_stop_tokens = list()

    for token in tokens:
        if token.isalpha():
            clean_tokens.append(token.strip("'").lower())
        else:
            continue

    for token in clean_tokens:
        if token not in stop_words:
            clean_no_stop_tokens.append(token)

    return clean_no_stop_tokens



def main():
    docs_path = os.path.join(os.getcwd(), "texts.txt")
    docs = read_docs(docs_path)
    for doc in docs:
        tokens = tokenize_docs(docs)
        tokens = clean_tokens(tokens)
        print(tokens)


if __name__ == '__main__':
    main()