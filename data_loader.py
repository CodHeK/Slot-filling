import pickle 

def load(filename):
    f = open('data/' + filename, 'rb')
    try:
        return pickle.load(f)
    except UnicodeDecodeError:
        return pickle.load(f, encoding='latin1')