from data_loader import DataLoader
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv1D, MaxPooling1D

from metrics.accuracy import conlleval
from process import Process

import argparse, os.path, json

import numpy as np

def train():
    with open('embeddings/word_embeddings.json', 'r') as f:
        embeddings = json.load(f)

    dataLoader = DataLoader()
    train_set, valid_set, _ = dataLoader.load('atis.pkl')

    w2idx, la2idx = embeddings['w2idx'], embeddings['la2idx']
    idx2w, idx2la = embeddings['idx2w'], embeddings['idx2la'] 

    n_classes = len(idx2la)
    n_vocab = len(idx2w)

    train_x, _, train_label = train_set
    val_x, _, val_label = valid_set

    words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
    groundtruth_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]

    words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
    groundtruth_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]

    print("Done processing word embeddings ...")
     
    model = Sequential()
    model.add(Embedding(n_vocab,100))
    model.add(Conv1D(128, 5, padding="same", activation='relu'))
    model.add(Dropout(0.25))
    model.add(GRU(100,return_sequences=True))
    model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
    model.compile('rmsprop', 'categorical_crossentropy')

    ### Training
    n_epochs = 100

    process = Process(model) # For Training and Testing

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


def loadEmbeddings():
    dataLoader = DataLoader()

    _, _, dicts = dataLoader.load('atis.pkl')
    w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']

    idx2w  = { w2idx[k]:k for k in w2idx }
    idx2la = { la2idx[k]:k for k in la2idx }

    embeddings = {
        "idx2w" : idx2w,
        "idx2la" : idx2la,
        "w2idx" : w2idx,
        "la2idx" : la2idx 
    }

    with open('embeddings/word_embeddings.json', 'w') as f:
        json.dump(embeddings, f)
    
    print("Word Embeddings Loaded ...")

def process_sentances(sentances):
    sentances = list(filter(lambda sentance : len(sentance) != 0, sentances))

    if len(sentances) > 1:
        last_sentance = sentances[-1]
        sentances = [ sentance[:-1] for sentance in sentances[:-1] ]
        sentances.append(last_sentance)

    return sentances

def test():
    process = Process()

    # Load trained model
    process.load('trained_model')

    f = open('test_sentances.txt', 'r')
    sentances = f.readlines()
    f.close()

    f = open('slots.txt', 'w')

    # Clean loaded sentances from file - removing '\n' from each sentance
    sentances = process_sentances(sentances)

    for sentance in sentances:
        BIO = process.test(sentance)# Test on sentance
        f.write(str(sentance) + "\n\n")

        # Print Slots to file
        for idx, slot in enumerate(BIO):
            if slot != 'O':
                f.write(str(sentance.split(" ")[idx]) + " - " + str(slot) + "\n")

        partition = ''
        for i in range(100):
            partition += '-'

        f.write(str(partition) + "\n")

    f.close()
    print("Output can be found in `slots.txt` file!")


if __name__ == '__main__':
    # FLAGS
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", help="trains model and validates", action="store_true")
    parser.add_argument("--test", help="tests model on an example input sentance", action="store_true")

    args = parser.parse_args()

    # Check if embeddings exists
    if not os.path.isfile('embeddings/word_embeddings.json'):
        loadEmbeddings()

    if args.train:
        train()
    
    if args.test:
        test()

            
            
