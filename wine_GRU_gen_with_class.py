import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time

for dirname, _, filenames in os.walk(os.getcwd()):
    for filename in filenames:
        if "first150k" in filename:
            path = os.path.join(dirname, filename)
            
reviews = pd.read_csv(path)
reviews = reviews[['country', 'description']][:100]
reviews.description = reviews.description.str.lower().str.replace('[^.,a-zA-Z\']+', ' ').str.split()
device = torch.device("cuda")
torch.cuda.empty_cache

def get_nclass_df(data_df, n_classes=3):
    n_country_reviews = pd.DataFrame(columns=['country', 'description'])
    top_countries = data_df.country.value_counts()[:n_classes].keys()
    for country in top_countries:
        country_reviews = pd.DataFrame(data_df[data_df.country == country])
        n_country_reviews = n_country_reviews.append(country_reviews)
    return n_country_reviews

reviews = get_nclass_df(reviews, 4)

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
        self.mask_token = mask_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token
        
        self.mask_index = self.add_token(self.mask_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)
        
    def lookup(self, token):
        return self._token_to_idx[token]
        
class ReviewVectorizer(object):
    
    def __init__(self, review_vocab, country_vocab, max_review_length):
        
        self.review_vocab = review_vocab
        self.country_vocab = country_vocab
        self.seq_length = max_review_length + 2
    
    def vectorize(self, review, vector_length=-1):
        indices = [self.review_vocab.begin_seq_index]
        indices.extend(self.review_vocab.lookup(token) for token in review)
        indices.append(self.review_vocab.end_seq_index)
        
        if vector_length < 0:
            vector_length = len(indices)
            
        from_vector = np.zeros(vector_length)
        from_vector[:len(indices)] = indices
        from_vector[len(indices):] = self.review_vocab.mask_index
        
        to_vector = np.zeros(vector_length)
        to_indices = indices[1:]
        to_vector[:len(to_indices)] = to_indices
        to_vector[len(to_indices):] = self.review_vocab.mask_index
        return from_vector, to_vector
        
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
    
vectorizer = ReviewVectorizer.from_df(reviews)

class WineDataset(Dataset):
    
    def __init__(self, review_df, vectorizer):
        self.review_df = review_df
        self._vectorizer = vectorizer
    
    def get_vectorizer(self):
        return self._vectorizer
    
    def __len__(self):
        return len(self.review_df)
    
    def __getitem__(self, index):
        row = self.review_df.iloc[index]
        from_vector, to_vector = self._vectorizer.vectorize(row.description, self._vectorizer.seq_length)
        country_index = self._vectorizer.country_vocab.lookup(row.country)
        return from_vector, to_vector, country_index
    
reviews = WineDataset(reviews, vectorizer)

class ReviewGenerationModel(nn.Module):
    
    def __init__(self, embed_size, vocab_size, class_vocab_size, hidden_size):
        super(ReviewGenerationModel, self).__init__()
        self.seq_embed = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=embed_size)
        self.class_embed = nn.Embedding(num_embeddings=class_vocab_size,
                                       embedding_dim=hidden_size)
        self.rnn = nn.GRU(input_size=embed_size,
                          hidden_size=hidden_size,
                          batch_first=True)
        self.decoder = nn.Linear(in_features=hidden_size,
                           out_features=vocab_size)
        
    def forward(self, x_in, class_index):
        x_embedded = self.seq_embed(x_in)
        class_embedded = self.class_embed(class_index)
        class_embedded = class_embedded.unsqueeze(0)
        output, _ = self.rnn(x_embedded, class_embedded)
        output = output.contiguous().view(-1, hidden_size)
        output = self.decoder(output)
        return output
        
vocab_size = len(vectorizer.review_vocab)
class_vocab_size = len(vectorizer.country_vocab)
mask_index = vectorizer.review_vocab.mask_index
embed_size = 8
hidden_size = 8
learning_rate = 0.01
batch_size = 32
epochs = 5

model = ReviewGenerationModel(embed_size=embed_size,
                                   vocab_size=vocab_size,
                                   class_vocab_size=class_vocab_size,
                                   hidden_size=hidden_size).to(device)
criterion = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
train_losses = list()
for epoch in range(epochs):
    start_time = time.time()
    losses = list()
    total = 0
    for review, target, country_idx in DataLoader(reviews, batch_size=batch_size):
        model.zero_grad()
        loss = 0
        review = torch.stack([rev for rev in review]).long().to(device)
        country_idx = torch.LongTensor(country_idx).to(device)
        output = model(review, country_idx).to(device)
        target = target.view(-1).long().to(device)
        loss += criterion(output, target, ignore_index=mask_index).to(device)
        loss.backward()
        optimizer.step()
        avg_loss = loss.item() / review.size(1)
        losses.append(avg_loss)
        total += 1
    epoch_loss = sum(losses) / total
    train_losses.append(epoch_loss)
    secs = int(time.time() - start_time)
    mins = secs // 60
    secs = secs % 60
    print('Train loss {:.3f} in {}minutes {}seconds'.format(epoch_loss, mins, secs))
    
model.eval()
gen_length = vectorizer.seq_length
class_idx = torch.randint(0, class_vocab_size, (1,))
temp = 0.8
text = ''
seed = 'wine'
with torch.no_grad():
    last_token = vectorizer.review_vocab.lookup(seed)
    class_idx = class_idx.long()
    for _ in range(gen_length):
        last_token = torch.LongTensor([[last_token]]).to(device)
        class_idx = class_idx.to(device)
        output = model(last_token, class_idx)
        distr = output.squeeze().div(temp).exp()
        guess = torch.multinomial(distr, 1).item()
        last_token = guess
        if guess:
            text += vectorizer.review_vocab.lookup_index(guess)
        else:
            break
        text += ' '
    print(text)
