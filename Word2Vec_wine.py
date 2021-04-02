import numpy as np
import pandas as pd
import os
import time
from gensim.models.word2vec import Word2Vec

t0 = time.perf_counter()

data_path = '/Users/ksu/projects/data/winemag-data_first150k.csv'
reviews = pd.read_csv(data_path)
reviews = reviews.description
token_list = [descr.split() for descr in reviews]


num_features = 300
min_word_count = 3
num_workers = 10
window_size = 5
subsampling = 1e-3

model = Word2Vec(token_list, workers=num_workers, vector_size=num_features, min_count=min_word_count,
                 window=window_size, sample=subsampling)

print(model.wv.most_similar('taste'))

elapsed = time.perf_counter() - t0
print(elapsed)
