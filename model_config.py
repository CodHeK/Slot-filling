from keras_contrib.losses import crf_loss

class Config:
    EMBEDDING_SIZE = 100
    HIDDEN_UNITS = 100
    DROPOUT = 0.5
    N_EPOCHS = 20
    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'rmsprop'
    MODEL = 'BLSTM'
    DATA_FILE = 'ner_dataset.csv'
    EMBEDDINGS_FILE = 'word_embeddings.json'
