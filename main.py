from data_loader import DataLoader
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from metrics.accuracy import conlleval
import numpy as np
import progressbar

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

train_f_scores = []
val_f_scores = []
best_val_f1 = 0

for i in range(n_epochs):
    print("Epoch {}".format(i))
    
    print("Training =>")
    train_pred_label = []
    avgLoss = 0

    bar = progressbar.ProgressBar(maxval=len(train_x))
    for idx, sentance in bar(enumerate(train_x)):
        label = train_label[idx]

        # one hot encode
        label = to_categorical(label, num_classes=n_classes)

        # pass as a batch
        sentance = sentance[np.newaxis,:] # makes [1, 2] -> [[1, 2]]

        label = label[np.newaxis, :] 
        
        if sentance.shape[1] > 1:
            loss = model.test_on_batch(sentance, label)
            avgLoss += loss

        pred = model.predict_on_batch(sentance)
        pred = np.argmax(pred,-1)[0]
        train_pred_label.append(pred)

    
    avgLoss = avgLoss/idx
    
    predword_train = [ list(map(lambda x: idx2la[x], y)) for y in train_pred_label]
    con_dict = conlleval(predword_train, groundtruth_train, words_train, 'r.txt')
    train_f_scores.append(con_dict['f1'])
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))

    print("Validating =>")
    
    val_pred_label = []
    avgLoss = 0
    
    bar = progressbar.ProgressBar(maxval=len(val_x))
    for idx, sentance in bar(enumerate(val_x)):
        label = val_label[idx]

        # one hot encode
        label = to_categorical(label, num_classes=n_classes)

        # pass as a batch
        sentance = sentance[np.newaxis,:] # makes [1, 2] -> [[1, 2]]

        label = label[np.newaxis, :] 
        
        if sentance.shape[1] > 1:
            loss = model.test_on_batch(sentance, label)
            avgLoss += loss
        
        pred = model.predict_on_batch(sentance)
        pred = np.argmax(pred,-1)[0]
        val_pred_label.append(pred)

    avgLoss = avgLoss/idx
    
    predword_val = [ list(map(lambda x: idx2la[x], y)) for y in val_pred_label]
    con_dict = conlleval(predword_val, groundtruth_val, words_val, 'r.txt')
    val_f_scores.append(con_dict['f1'])
    
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))

    if con_dict['f1'] > best_val_f1:
    	best_val_f1 = con_dict['f1']
    	open('model_architecture.json','w').write(model.to_json())
    	model.save_weights('best_model_weights.h5',overwrite=True)
    	print("Best validation F1 score = {}".format(best_val_f1))
    
     



