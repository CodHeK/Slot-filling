import logging
import sys, os

APP_PATH = str(os.path.dirname(os.path.realpath('../../logger.py')))
sys.path.append(APP_PATH)

from model_config import Config

def log(message, display=True):
    if display:
        print(message)
    LOG_FORMAT = '%(asctime)s - %(message)s'
    logging.basicConfig(filename='logs/model_' + str(Config.N_EPOCHS) + '_' + str(Config.MODEL) + '.log', 
                        filemode='a',
                        level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt='%d-%b-%y %H:%M:%S')

    logging.info(message)
