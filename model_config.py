from keras_contrib.losses import crf_loss

class Config:
    EMBEDDING_SIZE = 100
    DROPOUT = 0.25
    N_EPOCHS = 20
    LOSS = crf_loss
    OPTIMIZER = 'rmsprop'
    MODEL = 'GRU_CRF'
    DATA_FILE = 'atis.pkl'
    PORT = '9009'
