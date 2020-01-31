import logging
from .. import Config # need to import Config class in model_config.py

def log(message):
    LOG_FORMAT = '%(asctime)s - %(message)s'
    logging.basicConfig(filename='logs/model_' + str(Config.N_EPOCHS) + '_' + str(Config.MODEL) + '.log', 
                        filemode='w',
                        level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt='%d-%b-%y %H:%M:%S')

    logging.info(message)
