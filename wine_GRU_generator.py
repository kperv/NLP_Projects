import torch
import torch.nn as nn
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

torch.cuda.empty_cache()           
device = torch.device("cuda")
reviews = pd.read_csv(path)
reviews = reviews.description.str.lower().str.replace('[^a-zA-Z\']+', ' ').str.split()
#reviews = reviews[:10000]
reviews = [review for review in reviews]
max_length = max([len(review) for review in reviews])
reviews = [review + [0]*(max_length - len(review)) for review in reviews]

batch_size = 64
hidden_size = 16
embed_dim = 32
n_layers = 2
learning_rate = 0.01
epochs = 10

class Sequences(Dataset):
    def __init__(self, reviews):
        self.reviews = reviews
        self.token2idx = self.create_dict(self.reviews)
        self.vocab_size = len(self.token2idx)
        self.idx2token = {idx: token for token,idx in self.token2idx.items()}
        self.encoded_reviews = self.encode_reviews(self.reviews)
        
    def create_dict(self, reviews):
        all_tokens = [token for review in reviews for token in review ]
        unique_tokens = {token for token in all_tokens}
        return {token: idx for idx, token in enumerate(unique_tokens)}

    def encode_review(self, review):
        return list((self.token2idx[token] for token in review))
    
    def encode_reviews(self, reviews):
        return np.array([self.encode_review(review)for review in reviews])
    
    def __getitem__(self, index):
        return self.encoded_reviews[index, :-1], self.encoded_reviews[index, 1:]
    
    def __len__(self):
        return len(self.reviews)
    
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=embed_dim, hidden_size=hidden_size, n_layers=n_layers, device=device):
        super(RNN, self).__init__()
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.device=device
        self.encoder = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers=n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
    def init_hidden(self, batch_size):
        return torch.randn(self.n_layers, batch_size, self.hidden_size).to(self.device)
    
    def forward(self, x_in):
        encoded = self.encoder(x_in)
        output, _ = self.rnn(encoded.unsqueeze(1))
        output = self.decoder(output.squeeze(1))
        return output
        
dataset = Sequences(reviews)
model = RNN(vocab_size=dataset.vocab_size, device=device).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
train_losses = list()
for epoch in range(epochs):
    losses = list()
    start_time = time.time()
    total = 0
    for review, target in DataLoader(dataset, batch_size=batch_size):
        model.zero_grad()
        loss = 0
        for word_index in range(review.size(1)):
            output = model(review[:, word_index].long().to(device))
            loss += criterion(output, target[:, word_index].long().to(device))
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
    print('Epoch ', epoch)
    print('Train loss {:.3f} in {}minutes {}seconds'.format(epoch_loss, mins, secs))
    
model.eval()
gen_length = max_length - 10
temp = 1.0
text = ''
seed = 'wine'
with torch.no_grad():
    last_token = dataset.token2idx[seed]
    for _ in range(gen_length):
        output = model(torch.LongTensor([last_token]).to(device))
        distr = output.squeeze().div(temp).exp()
        guess = torch.multinomial(distr, 1).item()
        last_token = guess
        if guess:
            text += dataset.idx2token[guess]
        else:
            break
        text += ' '
print(text)
