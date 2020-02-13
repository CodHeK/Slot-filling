from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
import sys, os, csv, progressbar
import string, os, json
import numpy as np
from embeddings.parser import parseATIS

APP_PATH = str(os.path.dirname(os.path.realpath('../../' + __file__)))
sys.path.append(APP_PATH)

from model_config import Config
from logs.logger import log
from data_loader import load

class CustomEmbedding:
	def __init__(self, config={}, re_train=False):
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
		self.sentences, _, _, self.train_set, self.valid_set, self.indexes = parseATIS('data/' + Config.DATA_FILE)

		if re_train == False:
			self.embeddings_filename = str(Config.DATA_FILE) + '_' + str(Config.WORD_EMBEDDINGS) + '_embeddings.txt'
			
			if Config.WORD_EMBEDDINGS != 'None':
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

		else: # Model is re-training
			pass

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
