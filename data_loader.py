import pickle

class DataLoader:
    def __init__(self):
        pass

    def load(self, filename):
        f = open('data/' + filename, 'rb')
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            return pickle.load(f, encoding='latin1')
        