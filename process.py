from keras.utils import to_categorical
from logs.logger import log
from model_config import Config
from keras_contrib.layers import CRF
from keras.models import load_model
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from utils.print_utils import highlight
import numpy as np
import spacy, copy

import progressbar, json

class Process:
    def __init__(self, model=None):
        self.model = model
        self.indexes = self.getIndexes()
        self.n_classes = len(self.indexes['idx2la'])
    

    def getIndexes(self):
        with open('embeddings/' + Config.INDEXES_FILE, 'r') as f:
            indexes = json.load(f)
        
        return indexes
    

    def save(self, name):
        self.model.save('trained_model/' + name + '.h5')
        
        log("Saved model to disk", display=False)
        highlight('green', 'Saved model to disk')


    def load(self, filename):
        custom_objects = {
                          'CRF': CRF,
                          'crf_loss': crf_loss,
                          'crf_viterbi_accuracy': crf_viterbi_accuracy
                         }

        saved_model = load_model('trained_model/' + filename, custom_objects=custom_objects)

        log("Loaded model from disk", display=False)
        highlight('white', 'Loaded model from disk')

        self.model = saved_model

        return saved_model


    def train(self, train_set):
        train_x, train_label = train_set
        bar = progressbar.ProgressBar(maxval=len(train_x))

        for idx, sentence in bar(enumerate(train_x)):
            label = train_label[idx]

            # one hot encoding
            label = to_categorical(label, num_classes=self.n_classes)

            # pass as a batch
            sentence = sentence[np.newaxis,:] # makes [1, 2] -> [[1, 2]] i.e to form batch

            label = label[np.newaxis, :] 
    
            self.model.train_on_batch(sentence, label)


    def validate(self, valid_set):
        val_x, val_label = valid_set
        bar = progressbar.ProgressBar(maxval=len(val_x))

        val_pred_label = []
        
        for idx, sentence in bar(enumerate(val_x)):
            label = val_label[idx]

            # one hot encoding 
            label = to_categorical(label, num_classes=self.n_classes)

            # pass as a batch
            sentence = sentence[np.newaxis,:] # makes [1, 2] -> [[1, 2]] i.e to form batch

            label = label[np.newaxis, :] 
    
            self.model.test_on_batch(sentence, label)

            pred = self.model.predict_on_batch(sentence)

            pred = np.argmax(pred,-1)[0]
            val_pred_label.append(pred)

        predword_val = [ list(map(lambda x: self.indexes['idx2la'][str(x)], y)) for y in val_pred_label ]

        return predword_val


    def test(self, sentence):
        tokenizer = spacy.load('en_core_web_sm')

        tokens = tokenizer(sentence)

        sentence = []
        for token in tokens:
            sentence.append(token.text.lower())

        words = copy.deepcopy(sentence)

        # Encode in the input sentence
        for i in range(len(sentence)):
            word = sentence[i]
            
            # Convert 20 -> DIGITDIGIT
            if word.isdigit():
                numlen = len(word)
                word = ''
                for _ in range(numlen):
                    word += 'DIGIT'

            if word not in self.indexes['w2idx']:
                sentence[i] = self.indexes['w2idx']['<UNK>']
            else:
                sentence[i] = self.indexes['w2idx'][word]

        sentence = np.asarray(sentence)
        sentence = sentence[np.newaxis,:]

        pred = self.model.predict_on_batch(sentence)
        pred = np.argmax(pred,-1)[0]

        pred_slots = [ self.indexes['idx2la'][str(idx)] for idx in pred ]

        return (words, pred_slots)

