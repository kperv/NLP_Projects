import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time
from collections import Counter

for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.endswith('first150k.csv'):
            path = os.path.join(root, file)

reviews = pd.read_csv(path)
reviews = reviews[['country', 'description']]
original_reviews_for_prediction = reviews[-3:]
original_reviews_for_prediction = [review for review in original_reviews_for_prediction.description]
reviews.description = reviews.description.str.lower().str.replace('[^a-zA-Z\']+', ' ').str.split()
reviews_for_prediction = reviews[-3:]
countries_for_prediction = [country for country in reviews_for_prediction.country]
reviews_for_prediction = [review for review in reviews_for_prediction.description]
reviews = reviews[:-3]

def get_nclass_df(data_df, n_classes=3):
    n_country_reviews = pd.DataFrame(columns=['country', 'description'])
    top_countries = data_df.country.value_counts()[:n_classes].keys()
    for country in top_countries:
        country_reviews = pd.DataFrame(data_df[data_df.country == country])
        n_country_reviews = n_country_reviews.append(country_reviews)
    return n_country_reviews

reviews = get_nclass_df(reviews, 2)

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


class Vocabulary(object):

    def __init__(self, token_to_idx={}, unk_token=False):
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        self._unk_token = unk_token
        self._unk_index = -1

    def add_token(self, token):
        if token not in self._token_to_idx:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

    def lookup(self, token):
        return self._token_to_idx.get(token, self._unk_index)

    def lookup_index(self, index):
        return self._idx_to_token[index]

    def get_vocab(self):
        return self._token_to_idx

    def __str__(self):
        return "<Vocabulary (size={})>".format(len(self))

    def __len__(self):
        return len(self._token_to_idx)


class ReviewVectorizer(object):

    def __init__(self, review_vocab, country_vocab, max_review_length):

        self.review_vocab = review_vocab
        self.country_vocab = country_vocab
        self.max_review_length = max_review_length

    def get_review_vocab(self):
        return self.review_vocab

    def get_country_vocab(self):
        return self.country_vocab

    def get_max_review_length(self):
        return self.max_review_length

    def vectorize(self, review):
        one_hot_matrix_size = (len(self.review_vocab), self.max_review_length)
        one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)
        for index, word in enumerate(review):
            word_index = self.review_vocab.lookup(word)
            one_hot_matrix[word_index][index] = 1
        return one_hot_matrix

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        word_count = Counter()
        for review in review_df.description:
            word_count.update(review)

        tokens = [word for word, count in word_count.items() if count > cutoff]
        vocab_token_to_idx = {token: index for index, token in enumerate(tokens)}
        country_token_to_idx = {country: index for index, country in enumerate(set(review_df.country))}

        review_vocab = Vocabulary(vocab_token_to_idx, unk_token="@")
        country_vocab = Vocabulary(country_token_to_idx, unk_token=False)

        max_review_length = 0
        for index, row in review_df.iterrows():
            max_review_length = max(max_review_length, len(row.description))

        return cls(review_vocab, country_vocab, max_review_length)


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
    def load_dataset_and_make_vectorizer(cls, review_df):
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

    def get_vectorizer(self):
        return self._vectorizer

    def __len__(self):
        return self._split_size

    def __getitem__(self, index):
        row = self._split_df.iloc[index]
        review_matrix = self._vectorizer.vectorize(row.description)
        country_index = self._vectorizer.country_vocab.lookup(row.country)
        return {'x_review': review_matrix, 'y_country': country_index}

reviews = WineDataset.load_dataset_and_make_vectorizer(reviews)
vectorizer = reviews.get_vectorizer()
review_vocab = vectorizer.review_vocab.get_vocab()


class ReviewClassifier(nn.Module):

    def __init__(self, initial_num_channels, num_classes, num_channels):
        super(ReviewClassifier, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels,
                      out_channels=num_channels,
                      kernel_size=3),
            nn.ELU())
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x_review, apply_softmax=False):
        features = self.convnet(x_review)
        #features = features.squeeze(dim=2)
        features = torch.sum(features, dim=2)
        prediction_vector = self.fc(features)
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)
        return prediction_vector

batch_size = 256
epochs = 5
num_channels = 64

device = torch.device("cuda")
classifier = ReviewClassifier(initial_num_channels=len(vectorizer.review_vocab),
                             num_classes=len(vectorizer.country_vocab),
                             num_channels=num_channels)
classifier = classifier.to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.005)

def generate_batch(batch):
    text = torch.tensor([entry['x_review'] for entry in batch], dtype=torch.float32)
    label = torch.tensor([entry['y_country'] for entry in batch], dtype=torch.long)
    return text, label

def train_func(train_df):
    train_loss = 0
    train_acc = 0
    data = DataLoader(train_df, batch_size = batch_size, shuffle=True, collate_fn=generate_batch)
    for i, (text, label) in enumerate(data):
        optimizer.zero_grad()
        text = text.to(device)
        label = label.to(device)
        output = classifier.forward(text)
        loss = loss_func(output, label)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == label).sum().item()
    return train_loss/len(train_df), train_acc/len(train_df)

def test_func(test_df):
    loss = 0
    acc = 0
    data = DataLoader(test_df, batch_size=batch_size, collate_fn=generate_batch)
    for text, label in data:
        text = text.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = classifier.forward(text)
            loss = loss_func(output, label)
            loss += loss.item()
            acc += (output.argmax(1) == label).sum().item()
    return loss/len(test_df), acc/len(test_df)

import time
for epoch in range(epochs):
    start_time = time.time()
    reviews.set_split("train")
    train_loss, train_acc = train_func(reviews)
    reviews.set_split("val")
    val_loss, val_acc = test_func(reviews)
    secs = int(time.time() - start_time)
    mins = secs // 60
    secs = secs % 60
    print('Epoch: {} | time in {} minutes, {} seconds'.format(epoch+1, mins, secs))
    print('\t Loss: {:.4f} (train)\t | \t Acc: {:.1f}%'.format(train_loss, train_acc*100))
    print('\t Loss: {:.4f} (validate)\t | \t Acc: {:.1f}%'.format(val_loss, val_acc*100))


def predict_country(review, classifier, vectorizer):
    vectorized_review = vectorizer.vectorize(review)
    vectorized_review = torch.tensor(vectorized_review).unsqueeze(0)
    vectorized_review = vectorized_review.to(device)
    result = classifier(vectorized_review, apply_softmax=True)
    probability_values, indices = result.max(dim=1)
    index = indices.item()
    predicted_country = vectorizer.country_vocab.lookup_index(index)
    probability_value = probability_values.item()
    return {'country': predicted_country,
            'probability': probability_value}


for index, review in enumerate(reviews_for_prediction):
    predict_dict = predict_country(review, classifier, vectorizer)
    print("Review: ", original_reviews_for_prediction[index])
    print("Country is ", countries_for_prediction[index])
    print("Net result is {} {}".format(predict_dict['country'], predict_dict['probability']))
