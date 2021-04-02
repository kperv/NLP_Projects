import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter
import os
import time
import pandas as pd
import numpy as np


class ReviewDataset(Dataset):
    def __init__(self, review_df, classes, cutoff=25):
        self.review_df = review_df
        self.classes = classes
        self.cutoff = cutoff
        self.text_vocab = self.get_text_vocab()
        self.label_vocab = self.get_label_vocab()
        self.split = ""
        self.split_df = self.review_df

    def get_label_vocab(self):
        return {c: i for i, c in enumerate(self.classes)}

    def get_text_vocab(self):
        word_count = Counter()
        for review in self.review_df.description:
            word_count.update(review)
        tokens = [word for word, count in word_count.items() if count > self.cutoff]
        return {token: i for i, token in enumerate(tokens)}

    def get_num_features(self):
        return len(self.text_vocab)

    def vectorize(self, data, vocab):
        one_hot_vector = np.zeros(len(vocab))
        for item in data:
            if item in vocab:
                index = vocab[item]
                one_hot_vector[index] = 1
        return one_hot_vector

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, index):
        row = self.split_df.iloc[index]
        text_vector = self.vectorize(row.description, self.text_vocab)
        label = self.label_vocab[row.country]
        return text_vector, label

    def set_split(self, split):
        if split in "train test val".split():
            self.split = split
            self.split_df = self.review_df[self.review_df.split == split]
        else:
            print("Invalid split name")
            return None

    def get_split(self):
        return self.split


class ReviewClassifier(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features,
                             out_features=hidden_dim)
        self.relu =  nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_dim,
                             out_features=num_classes)

    def forward(self, x_in):
        intermediate = self.fc1(x_in).squeeze()
        intermediate = self.relu(intermediate)
        y_out = self.fc2(intermediate).squeeze()
        return y_out

def get_nclass_df(data_df, n_classes=3):
    n_country_reviews = pd.DataFrame(columns=['country', 'description'])
    top_countries = data_df.country.value_counts()[:n_classes].keys()
    for country in top_countries:
        country_reviews = pd.DataFrame(data_df[data_df.country == country])
        n_country_reviews = n_country_reviews.append(country_reviews)
    return n_country_reviews, top_countries

def add_splits(data_df, classes, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    split_reviews = pd.DataFrame(columns=['country', 'description', 'split'])
    for country in classes:
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

def generate_batch(batch):
    text = torch.tensor([entry[0] for entry in batch], dtype=torch.float32)
    label = torch.tensor([entry[1] for entry in batch], dtype=torch.long)
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

hidden_dim = 1000
batch_size = 256
epochs = 5
n_classes = 4
cutoff = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dir_path = "/Users/ksu/projects/data"
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith('first150k.csv'):
            path = os.path.join(root, file)

reviews = pd.read_csv(path)
reviews = reviews[['country', 'description']]
reviews.description = reviews.description.str.lower().str.replace('[^a-zA-Z\']+', ' ').str.split()
reviews, classes = get_nclass_df(reviews, n_classes)
reviews = add_splits(reviews, classes)
reviews = ReviewDataset(reviews, classes, cutoff)
num_features = reviews.get_num_features()
classifier = ReviewClassifier(num_features=num_features, hidden_dim=hidden_dim, num_classes=n_classes)
classifier = classifier.to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.005)

for epoch in range(epochs):
    start_time = time.time()
    reviews.set_split("train")
    train_loss, train_acc = train_func(reviews)
    reviews.set_split("val")
    val_loss, val_acc = test_func(reviews)
    secs = int(time.time() - start_time)
    mins = secs // 60
    secs = secs % 60
    print('Epoch: {} | time in {} minutes, {} seconds'.format(epoch + 1, mins, secs))
    print('\t Loss: {:.4f} (train)\t | \t Acc: {:.1f}%'.format(train_loss, train_acc * 100))
    print('\t Loss: {:.4f} (validate)\t | \t Acc: {:.1f}%'.format(val_loss, val_acc * 100))

reviews.set_split("test")
test_loss, test_acc = test_func(reviews)
print('\t Loss: {:.4f}\t|\tAcc: {:.1f}%'.format(test_loss, test_acc*100))