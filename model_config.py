from keras_contrib.losses import crf_loss

class Config:
    EMBEDDING_SIZE = 100
    DROPOUT = 0.25
    N_EPOCHS = 20
    LOSS = crf_loss
    OPTIMIZER = 'adam'
    MODEL = 'GRU_CRF'
    DATA_FILE = 'atis.pkl'
    WORD_EMBEDDINGS = 'glove'
    PORT = '9009'
    # Dont change the line below ...
    FILE_PATTERN = str(N_EPOCHS) + '_' + str(MODEL) + '_' + str(WORD_EMBEDDINGS)
