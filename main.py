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
import time

from model_config import Config
from utils.print_utils import partition, highlight
from utils.files import getBestSavedModel, clean
from metrics.accuracy import conlleval
from process import Process
from logs.logger import log
from embeddings.custom import CustomEmbedding


def train():
    word_model = CustomEmbedding()

    train_set, valid_set, indexes = word_model.train_set, word_model.valid_set, word_model.indexes

    w2idx, la2idx = indexes['w2idx'], indexes['la2idx']
    idx2w, idx2la = indexes['idx2w'], indexes['idx2la'] 

    n_classes = len(idx2la)
    n_vocab = len(idx2w)

    train_x, train_label = train_set
    valid_x, valid_label = valid_set

    log("Processing word indexes... ")

    words_val = [ list(map(lambda x: idx2w[x], w)) for w in valid_x]
    groundtruth_val = [ list(map(lambda x: idx2la[x], y)) for y in valid_label]

    log("Done processing word indexes!")
    
    ###############################################################
    '''
        MODEL 
    '''
    model = Sequential()

    model.add(word_model.EmbeddingLayer())

    model.add(Conv1D(128, 5, padding="same", activation='relu'))

    # model.add(Dropout(Config.DROPOUT))

    # model.add(Bidirectional(LSTM(units=Config.EMBEDDING_SIZE, 
    #                             dropout=Config.DROPOUT,
    #                             recurrent_dropout=Config.DROPOUT,
    #                             kernel_initializer=he_normal(),  
    #                             return_sequences=True)))
    
    # model.add(LSTM(units=Config.EMBEDDING_SIZE * 2, 
    #             return_sequences=True, 
    #             dropout=Config.DROPOUT, 
    #             recurrent_dropout=Config.DROPOUT,
    #             kernel_initializer=he_normal()))

    model.add(GRU(units=Config.EMBEDDING_SIZE, 
                  dropout=Config.DROPOUT, 
                  recurrent_dropout=Config.DROPOUT,
                  kernel_initializer=he_normal(),
                  return_sequences=True))

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
            process.save('trained_model_' + str(Config.N_EPOCHS) + '_' + str(Config.MODEL) + '_' + str(max_f1))
            log("New model saved!", display=False)

    highlight('white', 'Best validation F1 score : ' + str(max_f1))
    log('Best validation F1 score : ' + str(max_f1), display=False)

    log('Cleaning /trained_model folder...')
    clean()
    log('Removed all other saved models, kept the best model only!')


def process_sentances(sentances):
    sentances = [ sentance.strip('\n') for sentance in sentances ]

    sentances = list(filter(lambda sentance : len(sentance) != 0, sentances))

    return sentances

def test(process=None, sentences=None, read_file=True):
    start = time.time()
    _, best_model_filename, _ = getBestSavedModel()

    if read_file:
        process = Process()

        # Load trained model
        process.load(best_model_filename)

        f = open('tests/test_sentances.txt', 'r')
        sentences = f.readlines()
        f.close()

        # Clean loaded sentances from file - removing '\n' from each sentance
        sentences = process_sentances(sentences)

    f = open('tests/slots_' + str(Config.N_EPOCHS) + '_' + str(Config.MODEL) + '.txt', 'w')

    arr_slots = []
    for sentence in sentences:
        words, BIO = process.test(sentence) # Test on sentance
        f.write(str(sentence) + "\n\n")

        slots = {}
        slot_type = None
        value = ''
        prev_slot_type = None

        for idx, slot in enumerate(BIO):
            if slot != 'O':
                f.write(str(words[idx]) + " - " + str(slot) + "\n")

                '''
                    Grouping the slots

                    san - B-toloc.city_name
                    francisco - I-toloc.city_name
                    ------------------------------
                    Returns -> {'toloc.city_name': ['san francisco']}
                '''
                slot_type = slot.split("-")[1]
                pos = slot.split("-")[0]

                if pos == 'B':
                    if slot_type not in slots:
                        if prev_slot_type is not None:
                            slots[prev_slot_type].append(value.strip())
 
                        value = words[idx] + ' '
                        slots[slot_type] = []
                        prev_slot_type = slot_type
                    else:
                        slots[prev_slot_type].append(value.strip())
                        value = words[idx] + ' '
                else: # pos == 'I'
                    value += words[idx] + ' '

        slots[slot_type].append(value.strip())

        log('Slots compiled into groups...')

        f.write(partition(80) + "\n")

        arr_slots.append(slots)
        end = time.time()

    f.close()
    highlight('green', 'Output can be found in `slots.txt` file!')

    response_time = end - start
    if not read_file:
        return (response_time, arr_slots[0]) # As we're sending only one sentance in API URL

def model_params():
    log(
        '\n\n' +
        'MODEL PARAMETERS :' + '\n\n' +
        'EMBEDDING_SIZE = ' + str(Config.EMBEDDING_SIZE) + '\n' +  
        'DROPOUT = ' + str(Config.DROPOUT) + '\n' + 
        'N_EPOCHS = ' + str(Config.N_EPOCHS) + '\n' +
        'LOSS = ' + str(Config.LOSS) + '\n' +
        'OPTIMIZER = ' + str(Config.OPTIMIZER) + '\n'
        'MODEL = ' + str(Config.MODEL) + '\n'
        'DATA_FILE = ' + str(Config.DATA_FILE) + '\n'
        'PORT = ' + str(Config.PORT) + 
        '\n\n'
        )


if __name__ == '__main__':
    # FLAGS
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", help="trains model and validates", action="store_true")
    parser.add_argument("--test", help="tests model on an example input sentance", action="store_true")

    args = parser.parse_args()

    if args.train:
        log('*** TRAINING ***' + '\n')
        log(model_params())
        highlight('violet', 'Please open `logs/model.log` for all the logging information about the model')

        train()
    
    if args.test:
        log('*** TESTING ***' + '\n')

        test()

            
            

