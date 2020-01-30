from keras.utils import to_categorical
from keras.models import model_from_json
import numpy as np
import progressbar, json

class Process:
    def __init__(self, model=None):
        self.model = model
        self.embeddings = self.getEmbeddings()
        self.n_classes = len(self.embeddings['idx2la'])
    
    def getEmbeddings(self):
        with open('embeddings/word_embeddings.json', 'r') as f:
            embeddings = json.load(f)
        
        return embeddings
    
    def save(self, name):
        model_json = self.model.to_json()
        with open(name + ".json", "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(name + "_weights.h5")
        print("Saved model to disk")

    def load(self, filename):
        saved_model_file = open(filename + '.json', 'r')
        saved_model_json = saved_model_file.read()
        saved_model_file.close()

        saved_model = model_from_json(saved_model_json)

        saved_model.load_weights(filename + '_weights.h5')
        print("Loaded model from disk")

        self.model = saved_model

        return saved_model

    def train(self, train_set):
        train_x, _, train_label = train_set
        bar = progressbar.ProgressBar(maxval=len(train_x))

        avgLoss = 0

        for idx, sentance in bar(enumerate(train_x)):
            label = train_label[idx]

            # one hot encoding
            label = to_categorical(label, num_classes=self.n_classes)

            # pass as a batch
            sentance = sentance[np.newaxis,:] # makes [1, 2] -> [[1, 2]] i.e to form batch

            label = label[np.newaxis, :] 
    
            avgLoss += self.model.train_on_batch(sentance, label)

        return avgLoss/idx

    def validate(self, valid_set):
        val_x, _, val_label = valid_set
        bar = progressbar.ProgressBar(maxval=len(val_x))

        val_pred_label = []
        avgLoss = 0

        for idx, sentance in bar(enumerate(val_x)):
            label = val_label[idx]

            # one hot encoding 
            label = to_categorical(label, num_classes=self.n_classes)

            # pass as a batch
            sentance = sentance[np.newaxis,:] # makes [1, 2] -> [[1, 2]] i.e to form batch

            label = label[np.newaxis, :] 
    
            avgLoss += self.model.test_on_batch(sentance, label)

            pred = self.model.predict_on_batch(sentance)

            pred = np.argmax(pred,-1)[0]
            val_pred_label.append(pred)

        predword_val = [ list(map(lambda x: self.embeddings['idx2la'][x], y)) for y in val_pred_label ]

        return (self.model, predword_val, avgLoss/(idx+1))

    def test(self, sentance):
        sentance = sentance.split(" ")

        # Encode in the input sentance
        for i in range(len(sentance)):
            word = sentance[i]
            if word not in self.embeddings['w2idx']:
                sentance[i] = self.embeddings['w2idx']['<UNK>']
            else:
                sentance[i] = self.embeddings['w2idx'][word]

        sentance = np.asarray(sentance)
        sentance = sentance[np.newaxis,:]
 
        pred = self.model.predict_on_batch(sentance)
        pred = np.argmax(pred,-1)[0]

        pred_slots = [ self.embeddings['idx2la'][str(idx)] for idx in pred ]

        return pred_slots

