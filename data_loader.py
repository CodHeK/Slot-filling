import pickle 

def load(filename):
    f = open('data/' + filename, 'rb')
    try:
        return pickle.load(f)
    except UnicodeDecodeError:
        return pickle.load(f, encoding='latin1')


# train, test, embs = load('atis.pkl')

# for k, v in embs['labels2idx'].items():
#     print(k)
#     print("\n")