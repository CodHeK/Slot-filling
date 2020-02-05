from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv1D, LSTM, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.initializers import he_normal
from data_loader import load
import argparse, os.path, json, logging
import numpy as np

from model_config import Config
from utils.print_utils import partition, highlight
from metrics.accuracy import conlleval
from process import Process
from logs.logger import log
from embeddings.generate_embs import CreateEmbeddingsAndSets


def train(train_set, valid_set, embeddings):
    w2idx, la2idx = embeddings['w2idx'], embeddings['la2idx']
    idx2w, idx2la = embeddings['idx2w'], embeddings['idx2la'] 

    n_classes = len(idx2la)
    n_vocab = len(idx2w)

    train_x, train_label = train_set
    valid_x, valid_label = valid_set

    log("Processing word embeddings... ")

    words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
    groundtruth_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]

    words_val = [ list(map(lambda x: idx2w[x], w)) for w in valid_x]
    groundtruth_val = [ list(map(lambda x: idx2la[x], y)) for y in valid_label]

    log("Done processing word embeddings!")
    
    ###############################################################
    '''
        MODEL 
    '''
    model = Sequential()
    model.add(Embedding(n_vocab, Config.EMBEDDING_SIZE))

    model.add(Conv1D(128, 5, padding="same", activation='relu'))

    model.add(Dropout(Config.DROPOUT))

    # model.add(Bidirectional(LSTM(units=Config.HIDDEN_UNITS, 
    #                             dropout=Config.DROPOUT,
    #                             recurrent_dropout=Config.DROPOUT,
    #                             kernel_initializer=he_normal(),  
    #                             return_sequences=True)))

    model.add(GRU(100, return_sequences=True))

    model.add(TimeDistributed(Dense(n_classes, activation='softmax')))

    model.add(CRF(n_classes, sparse_target=False, learn_mode='join'))

    model.compile(Config.OPTIMIZER, Config.LOSS, metrics=[crf_viterbi_accuracy])

    ###############################################################

    process = Process(model)

    max_f1 = 0

    for i in range(Config.N_EPOCHS):
        log("Epoch " + str(i+1), display=False)
        highlight('violet', 'Epoch ' + str(i+1))

        partition(80)
        
        log("Training ")

        process.train(train_set)

        log("Validating ")

        predword_val = process.validate(valid_set)
    
        # Accuracy tests here using (predword_val, groundtruth_val, words_val) and save best model
        metrics = conlleval(predword_val, groundtruth_val, words_val, 'diff.txt')
        
        log('Precision = {}, Recall = {}, F1 = {}'.format(metrics['precision'], metrics['recall'], metrics['f1']))

        if metrics['f1'] > max_f1:
            max_f1 = metrics['f1']
            process.save('trained_model_' + str(Config.N_EPOCHS) + '_' + str(Config.MODEL))
            log("New model saved!", display=False)

    highlight('white', 'Best validation F1 score : ' + str(max_f1))


def loadEmbeddingsATIS():
    train_set, valid_set, dicts = load('atis.pkl') # load() from data_loader.py
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
    
    log("Word Embeddings Loaded ...")

    train_set = (train_set[0], train_set[2]) # packing only train_x and train_label
    valid_set = (valid_set[0], valid_set[2])

    return (train_set, valid_set, embeddings)


def loadEmbeddings():
    train_set, valid_set, embeddings = CreateEmbeddingsAndSets()

    # Used in process.py for loading embeddings
    with open('embeddings/' + Config.EMBEDDINGS_FILE, 'w') as f:
        json.dump(embeddings, f)
    
    log("Word Embeddings Dumped to JSON ...")

    return (train_set, valid_set, embeddings)


def process_sentances(sentances):
    sentances = [ sentance.strip('\n') for sentance in sentances ]

    sentances = list(filter(lambda sentance : len(sentance) != 0, sentances))

    return sentances

def test():
    process = Process()

    # Load trained model
    process.load('trained_model_' + str(Config.N_EPOCHS) + '_' + str(Config.MODEL))

    f = open('tests/test_sentances.txt', 'r')
    sentances = f.readlines()
    f.close()

    f = open('tests/slots_' + str(Config.N_EPOCHS) + '_' + str(Config.MODEL) + '.txt', 'w')

    # Clean loaded sentances from file - removing '\n' from each sentance
    sentances = process_sentances(sentances)

    for sentance in sentances:
        words, BIO = process.test(sentance) # Test on sentance
        f.write(str(sentance) + "\n\n")

        # Print Slots to file
        for idx, slot in enumerate(BIO):
            if slot != 'O':
                f.write(str(words[idx]) + " - " + str(slot) + "\n")

        f.write(partition(80) + "\n")

    f.close()
    highlight('green', 'Output can be found in `slots.txt` file!')

def model_params():
    log('MODEL PARAMETERS :' + '\n' +
        'EMBEDDING_SIZE = ' + str(Config.EMBEDDING_SIZE) + '\n' + 
        'HIDDEN_UNITS = ' + str(Config.HIDDEN_UNITS) + '\n' + 
        'DROPOUT = ' + str(Config.DROPOUT) + '\n' + 
        'N_EPOCHS = ' + str(Config.N_EPOCHS) + '\n' +
        'LOSS = ' + str(Config.LOSS) + '\n' +
        'OPTIMIZER = ' + str(Config.OPTIMIZER) + '\n'
        'MODEL = ' + str(Config.MODEL) + '\n'
        'DATA_FILE = ' + str(Config.DATA_FILE) + '\n'
        'EMBEDDINGS_FILE = ' + str(Config.EMBEDDINGS_FILE) + '\n')


if __name__ == '__main__':
    # FLAGS
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", help="trains model and validates", action="store_true")
    parser.add_argument("--test", help="tests model on an example input sentance", action="store_true")

    args = parser.parse_args()

    if args.train:
        ''' 
            Use this function if your dataset has the schema of type :

            sentance_idx | word | tag
        '''
        # train_set, valid_set, embeddings = loadEmbeddings()

        '''
            Else
        '''
        train_set, valid_set, embeddings = loadEmbeddingsATIS()

        log(model_params())
        highlight('violet', 'Please open `logs/model.log` for all the logging information about the model')
        train(train_set, valid_set, embeddings)
    
    if args.test:
        test()

            
            

