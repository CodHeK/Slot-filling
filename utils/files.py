from os import walk
import os, sys

APP_PATH = str(os.path.dirname(os.path.realpath('../../files.py')))
sys.path.append(APP_PATH)

from Slot_Filling.model_config import Config

def filesIn(path):
    files = []

    for _,_, filenames in walk(path):
        for filename in filenames:
            files.append(filename)


    return files

def getBestSavedModel():
    filenames = filesIn('/trained_model')

    max_acc = 0
    best_model_filename = None

    for filename in filenames:
        acc = filename.split("_")[-1][:-3]
        if acc[0].isdigit():
            if float(acc) > max_acc:
                max_acc = float(acc)
                best_model_filename = filename
    
    
    return (filenames, best_model_filename, max_acc)

def clean():
    filenames, best_filename, max_acc = getBestSavedModel()

    for filename in filenames:
        prefix = 'trained_model'
        suffix = filename.split("_")[-1]
        filename_len = len(filename)
        res = filename[len(prefix)+1:(filename_len - len(suffix) - 1)] # Ex: 20_GRU_CRF
        acc = suffix[:-3]

        if res == (str(Config.N_EPOCHS) + '_' + str(Config.MODEL)) and str(acc) != str(max_acc):
            os.system('cd /trained_model && rm %s' % filename)