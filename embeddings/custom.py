from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
import sys, os, csv, progressbar
import string, os, json
import numpy as np

APP_PATH = str(os.path.dirname(os.path.realpath('../../' + __file__)))
sys.path.append(APP_PATH)

from model_config import Config
from logs.logger import log
from data_loader import load

class CustomEmbedding:
	def __init__(self, config={}):
		'''
			from embeddings.custom import CustomEmbedding

			word_model = CustomEmbedding()

			# OR can add your own configuration else default values used.
			
			word_model = CustomEmbedding(config={
										'min_count': 5,
										'window': 5,
										'sg': 0,
										'pre_trained': True / False, # (default is False)
										'iter': 1000
										})

			model = Sequential()

			model.add(word_model.EmbeddingLayer())
		'''
		self.sentences, self.words, self.tags, self.train_set, self.valid_set, self.indexes = self.parseATIS('data/' + Config.DATA_FILE)

		self.embeddings_filename = str(Config.DATA_FILE) + '_' + str(Config.WORD_EMBEDDINGS) + '_embeddings.txt'
		
		if Config.WORD_EMBEDDINGS != None:
			self.min_count = config['min_count'] if 'min_count' in config else 5
			self.window = config['window'] if 'window' in config else 5
			self.sg = config['sg'] if 'sg' in config else 0
			self.iter = config['iter'] if 'iter' in config else 1000

			if Config.WORD_EMBEDDINGS == 'glove':
				self.pre_trained = True # Only option for now
			else:
				self.pre_trained = config['pre_trained'] if 'pre_trained' in config else False

			if self.pre_trained == False:
				log("Training Word2Vec word Embeddings...")
			else:
				log("Loading pre-trained " + str(Config.WORD_EMBEDDINGS) + " model...")
			
			# Check if word embeddings already built
			if not os.path.isfile('embeddings/' + self.embeddings_filename):
				self.model = self.load_model()
				self.save()
				log("Saved trained word embeddings at (embeddings/" + self.embeddings_filename + ")")
			else:
				log("Word Embeddings already exists...using exsiting file.")

		else: # Means using keras embedding layer
			pass

	def readCSV(self, filename):
		lines = []
		with open('data/' + filename, 'r', encoding='latin1') as csv_file:
			csv_data = csv.reader(csv_file, delimiter=',')
			for row in csv_data:
				lines.append(row)
		
		return lines[1:]

	def indexATIS(self):
		train_set, valid_set, dicts = load('atis.pkl') # load() from data_loader.py
		w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']

		idx2w  = { w2idx[k]:k for k in w2idx }
		idx2la = { la2idx[k]:k for k in la2idx }

		indexes = {
			"idx2w" : idx2w,
			"idx2la" : idx2la,
			"w2idx" : w2idx,
			"la2idx" : la2idx 
		}

		with open('embeddings/word_indexes.json', 'w') as f:
			json.dump(indexes, f)
			
		log("Word Indexes saved at (embeddings/word_indexes.json)...")

		train_x, _, train_label = train_set
		valid_x, _, valid_label = valid_set

		MAX_LEN = max(max([ len(s) for s in train_x ]), max([ len(s) for s in valid_x ]))

		# Add padding 
		train_x = pad_sequences(train_x, maxlen=MAX_LEN, padding='post', value=w2idx["<UNK>"])
		train_label = pad_sequences(train_label, maxlen=MAX_LEN, padding='post', value=la2idx["O"])

		valid_x = pad_sequences(valid_x, maxlen=MAX_LEN, padding='post', value=w2idx["<UNK>"])
		valid_label = pad_sequences(valid_label, maxlen=MAX_LEN, padding='post', value=la2idx["O"])

		train_set = (train_x, train_label) # packing only train_x and train_label
		valid_set = (valid_x, valid_label)

		return (train_set, valid_set, indexes)
	
	def parseATIS(self, filename):
		train_set, valid_set, indexes = self.indexATIS()

		# Convert indexed sentences in train_set & valid set to words usin w2idx
		sentences = []
		for indexed_sentence in train_set[0]: # train_set = train_x, train_label
			worded_sentence = []
			for w_idx in indexed_sentence:
				worded_sentence.append(indexes['idx2w'][w_idx])
		
			sentences.append(worded_sentence)
		
		for indexed_sentence in valid_set[0]: # valid_set = valid_x, valid_label
			worded_sentence = []
			for w_idx in indexed_sentence:
				worded_sentence.append(indexes['idx2w'][w_idx])
		
			sentences.append(worded_sentence)
		
		# Now dataset has all sentences in worded form which will go into Word2Vec to train
		return (sentences, None, None, train_set, valid_set, indexes)

	def index(sentences, words, tags):
		log("Creating Indexes")

		word2idx = { w : i for i, w in enumerate(words) }
		la2idx = { tag : i for i, tag in enumerate(tags) } # tag also called labels (la)

		idx2word = { v : k for k, v in word2idx.items() } 
		idx2la = { v : k for k, v in la2idx.items() }

		log("Spliting dataset into train and validation sets")

		MAX_LEN = max([ len(s) for s in sentences ])

		# w is of the form (word, label) 
		X = [[word2idx[w[0]] for w in s] for s in sentences]
		X = pad_sequences(X, maxlen=MAX_LEN, padding='post', value=w2idx["<UNK>"])

		y = [[la2idx[w[1]] for w in s] for s in sentences]
		y = pad_sequences(y, maxlen=MAX_LEN, padding='post', value=la2idx["O"])

		train_x, valid_x, train_label, valid_label = train_test_split(X, y, test_size=0.33)

		log("Done!")

		indexes = {
			"idx2w" : idx2word,
			"idx2la" : idx2la,
			"w2idx" : word2idx,
			"la2idx" : la2idx 
		}

		train_set = (train_x, train_label)
		valid_set = (valid_x, valid_label)

		return (train_set, valid_set, indexes)
	
	def parse(self, filename):
		'''
			Dataset schema:

			sentence_idx | word | tag 
	
		'''
		log("Parsing dataset ...")
		
		data = self.readCSV(filename)

		sentences = []
		sentence_number = 0
		sentence = []

		words = {}
		tags = {}

		bar = progressbar.ProgressBar(maxval=len(data))

		for line in bar(data):
			sentence_idx = line[0].split(":")[1].strip() if line[0] != '' else ''
			if str(sentence_number) != sentence_idx and sentence_idx != '':
				if len(sentence) > 0:
					sentences.append(sentence)
				sentence = []

			word, tag = line[1].lower(), line[-1]

			# Generating a set of words & tags
			if word not in words:
				words[word] = True
			if tag not in tags:
				tags[tag] = True

			sentence.append((word, tag))

		words['<UNK>'] = True

		train_set, valid_set, indexes = self.index(sentences, words, tags)

		return (sentences, words, tags, train_set, valid_set, indexes)


	def load_model(self):
		if self.pre_trained == True:
			if Config.WORD_EMBEDDINGS == 'word2vec':
				# Download here : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing (1.5G)
				filename = 'GoogleNews-vectors-negative300.bin'
				self.model = KeyedVectors.load_word2vec_format('embeddings/' + filename, binary=True)
			
			elif Config.WORD_EMBEDDINGS == 'glove':
				# Download here : http://nlp.stanford.edu/data/glove.6B.zip
				glove_input_file = 'glove.6B.100d.txt'
				word2vec_output_file = 'glove.6B.100d.txt.word2vec'
				if not os.path.isfile('embeddings/' + word2vec_output_file):
					glove2word2vec('embeddings/' + glove_input_file, 'embeddings/' + word2vec_output_file)
				self.model = KeyedVectors.load_word2vec_format('embeddings/' + word2vec_output_file, binary=False)
		
		else:
			# Know more here: https://radimrehurek.com/gensim/models/word2vec.html
			self.model = Word2Vec(self.sentences, size=Config.EMBEDDING_SIZE, min_count=self.min_count, 
						window=self.window, sg=self.sg, iter=self.iter)

		return self.model

	def save(self):
		self.model.wv.save_word2vec_format('embeddings/' + self.embeddings_filename, binary=False)

	def EmbeddingLayer(self):
		embedding_index = {}
		file = open(os.path.join('', 'embeddings/' + self.embeddings_filename), encoding='utf-8')
		for line in file:
			values = line.split()
			word = values[0]
			vec = np.asarray(values[1:])
			embedding_index[word] = vec
		file.close()

		# Convert the word embedding into tokenized vector
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(self.sentences)
		sequences = tokenizer.texts_to_sequences(self.sentences)

		max_len = max([ len(s) for s in sequences ])

		vocab_size = len(tokenizer.word_index) + 1

		padded = pad_sequences(sequences, maxlen=max_len)

		embedding_matrix = np.zeros(( vocab_size, Config.EMBEDDING_SIZE ))

		word_index = tokenizer.word_index

		for word, i in word_index.items():
			if i > vocab_size:
				continue

			embedding_vector = embedding_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector

		# Return keras embedding layer 
		return Embedding(vocab_size, 
						 Config.EMBEDDING_SIZE, 
						 embeddings_initializer = Constant(embedding_matrix),
						 trainable = False)
