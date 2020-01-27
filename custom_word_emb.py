from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
import string, os
import numpy as np

class WordEmbedding:
	def __init__(self, file, config):
		self.train_data = self.clean(file)

		self.size = config['size'] if 'size' in config else 100 
		self.min_count = config['min_count'] if 'min_count' in config else 5
		self.window = config['window'] if 'window' in config else 5
		self.sg = config['sg'] if 'sg' in config else 0
		self.pre_trained = config['pre_trained'] if 'pre_trained' in config else None

		self.model = None
	
	def clean(self, file):
		with open(file, 'r') as file:
			text = file.readlines()

		training_corpus = ""
		for lines in text:
			training_corpus += lines

		train_data = []
		sentences = training_corpus.split(".")

		for s in sentences:
			s = s.strip().replace("\n", " ").replace("\r", " ")			
			words = s.split(" ")

			for i in range(len(words)):
				new_word = ""
				for char in words[i]:
					if char not in string.punctuation:
						new_word += char
				words[i] = new_word.lower()
			
			train_data.append(words)

		return train_data
	
	def train(self):
		if self.pre_trained == 'word2vec':
			# Download here : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
			filename = 'GoogleNews-vectors-negative300.bin'
			self.model = KeyedVectors.load_word2vec_format(filename, binary=True)

		elif self.pre_trained == 'glove':
			# Download here : http://nlp.stanford.edu/data/glove.6B.zip
			glove_input_file = 'glove.6B.100d.txt'
			word2vec_output_file = 'glove.6B.100d.txt.word2vec'
			glove2word2vec(glove_input_file, word2vec_output_file)
			self.model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

		else:
			self.model = Word2Vec(self.train_data, size=self.size, min_count=self.min_count, 
						window=self.window, sg=self.sg)
		
		return self.model
	

	def save(self, filename):
		self.model.wv.save_word2vec_format(filename, binary=False)
	
	def buildEmbeddingLayer(self, filename, EMBEDDING_DIM=100):
		embedding_index = {}
		file = open(os.path.join('', filename), encoding='utf-8')
		for line in file:
			values = line.split()
			word = values[0]
			vec = np.asarray(values[1:])
			embedding_index[word] = vec
		file.close()

		# Convert the word embedding into tokenized vector

		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(self.train_data)
		sequences = tokenizer.texts_to_sequences(self.train_data)

		max_len = max([ len(s) for s in sequences ])

		vocab_size = len(tokenizer.word_index) + 1

		padded = pad_sequences(sequences, maxlen = max_len)

		embedding_matrix = np.zeros(( vocab_size, EMBEDDING_DIM))

		word_index = tokenizer.word_index

		for word, i in word_index.items():
			if i > vocab_size:
				continue
				
			embedding_vector = embedding_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector

		return Embedding(vocab_size, 
						 EMBEDDING_DIM, 
						 embeddings_initializer = Constant(embedding_matrix),
						 input_length = max_len,
						 trainable = False)




model = WordEmbedding('corpus.txt', config={ 'min_count': 1 })

embeddings = model.train()

model.save('test_save.txt')

embeddings_layer = model.buildEmbeddingLayer('test_save.txt')

print(embeddings_layer)