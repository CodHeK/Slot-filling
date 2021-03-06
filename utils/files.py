from os import walk
import os, sys

APP_PATH = str(os.path.dirname(os.path.realpath('../../' + __file__)))
sys.path.append(APP_PATH)

from model_config import Config

def getFilesIn(path):
    files = []

    for _,_, filenames in walk(path):
        for filename in filenames:
            files.append(filename)

    return files

def nameUtil(filename):
    ''' 
    Example:

        filename = 'trained_model_20_GRU_CRF_glove_84.4.h5'

        prefix = 'trained_model'

        res = '20_GRU_CRF_glove' == (Config.N_EPOCHS + Config.MODEL + Config.WORD_EMBEDDINGS) = Config.FILE_PATTERN

        suffix = '84.4.h5'

        f1 = '84.4'
    '''
    prefix = 'trained_model'
    suffix = filename.split("_")[-1]
    filename_len = len(filename)

    pattern = filename[len(prefix)+1:(filename_len - len(suffix) - 1)] # Ex: 20_GRU_CRF_glove
    f1 = suffix[:-3]

    return (pattern, f1)

def getBestSavedModelToTest():
    filenames = getFilesIn('trained_model')

    max_f1 = 0.0
    best_model_filename = None

    for filename in filenames:
        f1 = filename.split("_")[-1][:-3]
        if f1[0].isdigit():
            if float(f1) > max_f1:
                max_f1 = float(f1)
                best_model_filename = filename
    
    return best_model_filename

def getCurrentConfigBestModel():
    filenames = getFilesIn('trained_model')

    max_f1 = 0.0
    best_model_filename = None

    for filename in filenames:
        pattern, f1 = nameUtil(filename)
        if f1[0].isdigit() and pattern == str(Config.FILE_PATTERN):
            if float(f1) > max_f1:
                max_f1 = float(f1)
                best_model_filename = filename
    
    return (filenames, max_f1)


def clean():
    filenames, max_f1 = getCurrentConfigBestModel()

    for filename in filenames:
        pattern, f1 = nameUtil(filename)

        # Keep only the file with highest F1 Score
        if pattern == (str(Config.FILE_PATTERN)) and str(f1) != str(max_f1):
            os.system('cd trained_model/ && rm %s' % filename)
