import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dir_path = "/Users/ksu/projects/data"
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith('first150k.csv'):
            path = os.path.join(root, file)

reviews = pd.read_csv(path)
reviews = reviews[['country', 'description']][:1000]

def get_nclass_df(data_df, n_classes=3):
    n_country_reviews = pd.DataFrame(columns=['country', 'description'])
    top_countries = data_df.country.value_counts()[:n_classes].keys()
    for country in top_countries:
        country_reviews = pd.DataFrame(data_df[data_df.country == country])
        n_country_reviews = n_country_reviews.append(country_reviews)
    return n_country_reviews

reviews = get_nclass_df(reviews, 2)

original_reviews_for_prediction = reviews[-3:]
original_reviews_for_prediction = [review for review in original_reviews_for_prediction.description]
reviews.description = reviews.description.str.lower().str.replace('[^a-zA-Z\']+', ' ').str.split()
reviews_for_prediction = reviews[-3:]
countries_for_prediction = [country for country in reviews_for_prediction.country]
reviews_for_prediction = [review for review in reviews_for_prediction.description]

reviews = reviews[:-3]

class Vocabulary(object):

    def __init__(self, token_to_idx=None):
        if not token_to_idx:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

    def add_token(self, token):
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
            return index

    def lookup(self, token):
        return self._token_to_idx[token]

    def lookup_index(self, index):
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary (size={})>".format(len(self))

    def __len__(self):
        return len(self._token_to_idx)

class SequenceVocabulary(Vocabulary):

    def __init__(self, token_to_idx=None, mask_token="<MASK>",
                 begin_seq_token="<BEGIN>", end_seq_token="<END>"):
        super(SequenceVocabulary, self).__init__(token_to_idx)
        self._mask_token = mask_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

        def lookup(self, token):
            return self._token_to_idx[token]

class ReviewVectorizer(object):

    def __init__(self, review_vocab, country_vocab, max_seq_length):
        self.review_vocab = review_vocab
        self.country_vocab = country_vocab
        self.max_seq_length = max_seq_length

    def vectorize(self, review, vector_length=-1):
        indices = [self.review_vocab.begin_seq_index]
        indices.extend(self.review_vocab.lookup(token)
                       for token in review)
        indices.append(self.review_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.review_vocab.mask_index
        return out_vector, len(indices)

    @classmethod
    def from_df(cls, review_df):
        review_vocab = SequenceVocabulary()
        country_vocab = Vocabulary()
        max_review_length = 0
        for index, row in review_df.iterrows():
            for token in row.description:
                review_length = len(row.description)
                max_review_length = np.maximum(max_review_length, review_length)
                review_vocab.add_token(token)
            country_vocab.add_token(row.country)
        return cls(review_vocab, country_vocab, max_review_length)

def add_splits(data_df, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    split_reviews = pd.DataFrame(columns=['country', 'description', 'split'])
    for country in data_df.country.unique():
        country_reviews = pd.DataFrame(data_df[data_df.country == country])
        n_total = len(country_reviews)
        n_train = int(n_total * train_prop)
        n_val = int(n_total * val_prop)
        n_test = int(n_total * test_prop)

        country_reviews['split'] = None
        country_reviews.split.iloc[:n_train] = 'train'
        country_reviews.split.iloc[n_train:n_train+n_val] = 'val'
        country_reviews.split.iloc[n_train+n_val:] = 'test'

        split_reviews = split_reviews.append(country_reviews)
    return split_reviews

reviews = add_splits(reviews)

class WineDataset(Dataset):

    def __init__(self, review_df, vectorizer):
        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split == "train"]
        self.val_df = self.review_df[self.review_df.split == "val"]
        self.test_df = self.review_df[self.review_df.split == "test"]

        self.train_size = len(self.train_df)
        self.val_size = len(self.val_df)
        self.test_size = len(self.test_df)

        self._split_dict = {'train': (self.train_df, self.train_size),
                            'val': (self.val_df, self.val_size),
                            'test': (self.test_df, self.test_size)}
        self.set_split('train')

    def set_split(self, split):
        self._split = split
        self._split_df, self._split_size = self._split_dict[self._split]

    def get_split(self):
        return self._split

    @classmethod
    def make_vectorizer(cls, review_df):
        return cls(review_df, ReviewVectorizer.from_df(review_df))

    def get_vectorizer(self):
        return self._vectorizer

    def __len__(self):
        return self._split_size

    def __getitem__(self, index):
        row = self._split_df.iloc[index]
        review_vector, vec_length = self._vectorizer.vectorize(row.description, self._vectorizer.max_seq_length)
        country_index = self._vectorizer.country_vocab.lookup(row.country)
        return {'x_review': review_vector, 'y_country': country_index, 'x_length': vec_length}

reviews = WineDataset.make_vectorizer(reviews)
vectorizer = reviews.get_vectorizer()
embedding = nn.Embedding(len(vectorizer.review_vocab), 8)
print("Fist review: ", reviews[0])
print("First embedding : ", embedding[0])

