from data_loader import DataLoader
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv1D, MaxPooling1D

from metrics.accuracy import conlleval
from process import Process

import numpy as np

dataLoader = DataLoader()

train_set, valid_set, dicts = dataLoader.load('atis.pkl')
w2idx, labels2idx = dicts['words2idx'], dicts['labels2idx']

idx2w  = {w2idx[k]:k for k in w2idx}
idx2la = {labels2idx[k]:k for k in labels2idx}

n_classes = len(idx2la)
n_vocab = len(idx2w)

train_x, _, train_label = train_set
val_x, _, val_label = valid_set

words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
groundtruth_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]

words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
groundtruth_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]

model = Sequential()
model.add(Embedding(n_vocab,100))
model.add(Conv1D(128, 5, padding="same", activation='relu'))
model.add(Dropout(0.25))
model.add(GRU(100,return_sequences=True))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
model.compile('rmsprop', 'categorical_crossentropy')

### Training
n_epochs = 1

process = Process(model, n_classes, idx2la, w2idx) # For Training and Testing

minLoss = 1000000

for i in range(n_epochs):
    print("Epoch " + str(i))
    
    print("Training ")

    loss = process.train(train_set)

    print("Loss : " + str(loss))


    print("Validating ")

    model, predword_val, loss = process.validate(valid_set)

    if loss < minLoss:
        minLoss = loss
        process.save('trained_model')

    # Do Accuracy tests here using (predword_val, groundtruth_val, words_val) and save best model


print("Least Loss : " + str(minLoss))


process.load('trained_model')

# TEST 

sentance = 'I want to see all the flights from washington to berlin flying tomorrow'

print(process.test(sentance))