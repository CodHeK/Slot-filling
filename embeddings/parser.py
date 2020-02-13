from keras.preprocessing.sequence import pad_sequences
import sys, os, csv, progressbar
import string, os, json
import numpy as np

APP_PATH = str(os.path.dirname(os.path.realpath('../../' + __file__)))
sys.path.append(APP_PATH)

from model_config import Config
from logs.logger import log
from data_loader import load

def readCSV(filename):
    lines = []
    with open('data/' + filename, 'r', encoding='latin1') as csv_file:
        csv_data = csv.reader(csv_file, delimiter=',')
        for row in csv_data:
            lines.append(row)
    
    return lines[1:]

def indexATIS():
    train_set, valid_set, dicts = load('atis.pkl') # load() from data_loader.py
    w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']

    idx2w  = { w2idx[k]:k for k in w2idx }
    idx2la = { la2idx[k]:k for k in la2idx }

    indexes = {
        "idx2w" : idx2w,
        "idx2la" : idx2la,
        "w2idx" : w2idx,
        "la2idx" : la2idx 
    }

    with open('embeddings/word_indexes.json', 'w') as f:
        json.dump(indexes, f)
        
    log("Word Indexes saved at (embeddings/word_indexes.json)...")

    train_x, _, train_label = train_set
    valid_x, _, valid_label = valid_set

    MAX_LEN = max(max([ len(s) for s in train_x ]), max([ len(s) for s in valid_x ]))

    # Add padding 
    train_x = pad_sequences(train_x, maxlen=MAX_LEN, padding='post', value=w2idx["<UNK>"])
    train_label = pad_sequences(train_label, maxlen=MAX_LEN, padding='post', value=la2idx["O"])

    valid_x = pad_sequences(valid_x, maxlen=MAX_LEN, padding='post', value=w2idx["<UNK>"])
    valid_label = pad_sequences(valid_label, maxlen=MAX_LEN, padding='post', value=la2idx["O"])

    train_set = (train_x, train_label) # packing only train_x and train_label
    valid_set = (valid_x, valid_label)

    return (train_set, valid_set, indexes)

def parseATIS(filename):
    train_set, valid_set, indexes = indexATIS()

    # Convert indexed sentences in train_set & valid set to words usin w2idx
    sentences = []
    for indexed_sentence in train_set[0]: # train_set = train_x, train_label
        worded_sentence = []
        for w_idx in indexed_sentence:
            worded_sentence.append(indexes['idx2w'][w_idx])
    
        sentences.append(worded_sentence)
    
    for indexed_sentence in valid_set[0]: # valid_set = valid_x, valid_label
        worded_sentence = []
        for w_idx in indexed_sentence:
            worded_sentence.append(indexes['idx2w'][w_idx])
    
        sentences.append(worded_sentence)
    
    # Now dataset has all sentences in worded form which will go into Word2Vec to train
    return (sentences, None, None, train_set, valid_set, indexes)

def index(sentences, words, tags):
    log("Creating Indexes")

    word2idx = { w : i for i, w in enumerate(words) }
    la2idx = { tag : i for i, tag in enumerate(tags) } # tag also called labels (la)

    idx2word = { v : k for k, v in word2idx.items() } 
    idx2la = { v : k for k, v in la2idx.items() }

    log("Spliting dataset into train and validation sets")

    MAX_LEN = max([ len(s) for s in sentences ])

    # w is of the form (word, label) 
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(X, maxlen=MAX_LEN, padding='post', value=w2idx["<UNK>"])

    y = [[la2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(y, maxlen=MAX_LEN, padding='post', value=la2idx["O"])

    train_x, valid_x, train_label, valid_label = train_test_split(X, y, test_size=0.33)

    log("Done!")

    indexes = {
        "idx2w" : idx2word,
        "idx2la" : idx2la,
        "w2idx" : word2idx,
        "la2idx" : la2idx 
    }

    train_set = (train_x, train_label)
    valid_set = (valid_x, valid_label)

    return (train_set, valid_set, indexes)

def parse(filename):
    '''
        Dataset schema:

        sentence_idx | word | tag 

    '''
    log("Parsing dataset ...")
    
    data = readCSV(filename)

    sentences = []
    sentence_number = 0
    sentence = []

    words = {}
    tags = {}

    bar = progressbar.ProgressBar(maxval=len(data))

    for line in bar(data):
        sentence_idx = line[0].split(":")[1].strip() if line[0] != '' else ''
        if str(sentence_number) != sentence_idx and sentence_idx != '':
            if len(sentence) > 0:
                sentences.append(sentence)
            sentence = []

        word, tag = line[1].lower(), line[-1]

        # Generating a set of words & tags
        if word not in words:
            words[word] = True
        if tag not in tags:
            tags[tag] = True

        sentence.append((word, tag))

    words['<UNK>'] = True

    train_set, valid_set, indexes = index(sentences, words, tags)

    return (sentences, words, tags, train_set, valid_set, indexes)