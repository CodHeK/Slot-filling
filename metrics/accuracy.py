import numpy
import os, stat
from os import chmod
import random

def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open('metrics/' + filename,'w')
    f.writelines(out)
    f.close()
    
    return get_metrics(filename)

def get_metrics(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''

    if os.path.exists('metrics/' + filename):
        chmod('metrics/conlleval.pl', stat.S_IRWXU) # give the execute permissions
        cmd = 'cd metrics && ./conlleval.pl < %s | grep accuracy' % filename
        out = os.popen(cmd).read().split()
        # os.system('cd metrics && rm %s' % filename)  # delete file
        return {'precision': float(out[3][:-2]), 'recall': float(out[5][:-2]), 'f1': float(out[7])}