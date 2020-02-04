'''
Dataset schema:

sentance_idx | word | tag 

'''
# from future.utils import iteritems
import sys, os, csv, progressbar
from math import nan
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


APP_PATH = str(os.path.dirname(os.path.realpath('../../generate_embs.py')))
sys.path.append(APP_PATH)

from model_config import Config
from logs.logger import log

def readCSV(filename):
    lines = []
    with open('data/' + filename, 'r', encoding='latin1') as csv_file:
        csv_data = csv.reader(csv_file, delimiter=',')
        for row in csv_data:
            lines.append(row)
    
    return lines[1:]

def parse(filename):
    log("Parsing dataset ...")

    data = readCSV(filename)

    sentances = []
    sentance_number = 0
    sentance = []

    words = {}
    tags = {}

    bar = progressbar.ProgressBar(maxval=len(data))

    for line in bar(data):
        sentance_idx = line[0].split(":")[1].strip() if line[0] != '' else ''
        if str(sentance_number) != sentance_idx and sentance_idx != '':
            if len(sentance) > 0:
                sentances.append(sentance)
            sentance = []

        word, tag = line[1], line[-1]

        # Generating a set of words & tags
        if word not in words:
            words[word] = True
        if tag not in tags:
            tags[tag] = True

        sentance.append((word, tag))
    
    words['<UNK>'] = True

    return (sentances, words, tags)
    

def index(sentances, words, tags):
    log("Creating Indexes")

    word2idx = { w : i for i, w in enumerate(words) }
    la2idx = { tag : i for i, tag in enumerate(tags) } # tag also called labels (la)

    idx2word = { v : k for k, v in word2idx.items() } 
    idx2la = { v : k for k, v in la2idx.items() }

    log("Spliting dataset into train and validation sets")

    MAX_LEN = max([len(s) for s in sentances ])

    X = [[word2idx[w[0]] for w in s] for s in sentances]
    X = pad_sequences(X, maxlen=MAX_LEN, padding='post', value=len(words)-1)

    y = [[la2idx[w[1]] for w in s] for s in sentances]
    y = pad_sequences(y, maxlen=MAX_LEN, padding='post', value=la2idx["O"])
    
    train_x, valid_x, train_label, valid_label = train_test_split(X, y, test_size=0.33)

    log("Done!")

    embeddings = {
        "idx2w" : idx2word,
        "idx2la" : idx2la,
        "w2idx" : word2idx,
        "la2idx" : la2idx 
    }

    train_set = (train_x, train_label)
    valid_set = (valid_x, valid_label)

    return (train_set, valid_set, embeddings)

def CreateEmbeddingsAndSets():
    filename = Config.DATA_FILE
    sentances, words, tags = parse(filename)

    return index(sentances[:5000], words, tags)

