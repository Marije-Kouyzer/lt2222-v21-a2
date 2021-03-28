import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
import math
from sklearn.metrics import confusion_matrix as cfm

# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# Function for Part 1
def preprocess(inputfile):
	nltk.download('wordnet')
	lemmatizer = WordNetLemmatizer()
	l = inputfile.readlines()
	processed_text = []
	for i in l[1:]:
		j = i.split('\t')
		j[2] = j[2].lower()
		j[2] = lemmatizer.lemmatize(j[2])
		item = tuple(j)
		processed_text.append(item)
	return processed_text

# Code for part 2
class Instance:
	def __init__(self, neclass, features):
		self.neclass = neclass
		self.features = features

	def __str__(self):
		return "Class: {} Features: {}".format(self.neclass, self.features)

	def __repr__(self):
		return str(self)

def create_instances(data):
	instances = []
	for i in data:
		if i[4].startswith('B'):
			# neclass
			neclass = i[4].split('-')[1]
			neclass = neclass.strip('\n')
			
			# features
			features = []
			sentence_number = i[1]
			index_B = int(i[0])
			index = index_B
			while True:
				index += 1
				if not data[index][4].startswith('I'):
					break
			# index is now the index of the first word after the named entity
			
			# adding the features of the five words before the named entity
			c = 1
			while c <= 5:
				index_current = index_B - c
				if data[index_current][1] == data[index_B][1]:
					features.append(data[index_current][2])
					c += 1
				else:
					break
			if c < 6:
				length = len(features)
				amount = 5 - length
				for x in range(amount):
					features.append('<s>')
			features = features[::-1]
			
			# adding the features of the five words after the named entity
			d = 0
			while d <= 4:
				index_current = index + d
				if index_current == len(data): #in case the last ne does not have 5 words after it at all
					break
				if data[index_current][1] == data[index_B][1]:
					features.append(data[index_current][2])
					d += 1
				else:
					break
			if d < 5:
				length = len(features)
				amount = 10 - length
				for x in range(amount):
					features.append('<e>')
			
			instances.append(Instance(neclass, features))
	return instances

# Code for part 3
def create_table(instances):
	f = features_collect(instances)
	df = pd.DataFrame()
	df['neclass'] = [i.neclass for i in instances]
	for feature in f:
		l = []
		for i in instances:
			l.append(0)
			for word in i.features:
				if word == feature:
					l[len(l)-1] += 1
		df[feature] = l
	df = reduce_features(df, 1000)
	return df

def features_collect(instances):
	all_features = []
	for i in instances:
		for j in i.features:
			if not j in all_features:
				all_features.append(j)
	return all_features

def reduce_features(df, num_features):
	partial_df = df.iloc[:,1:]
	trunc = TruncatedSVD(n_components=num_features)
	trunc.fit(partial_df)
	reduced_df = trunc.transform(partial_df)
	reduced_df = pd.DataFrame(reduced_df)
	classes = df.iloc[:,0]
	reduced_df.insert(0, 'class', classes)
	return reduced_df

def ttsplit(bigdf):
	eighty_percent = math.floor(len(bigdf)*0.8)
	train_df = bigdf.sample(n=eighty_percent)
	test_df = bigdf.drop(train_df.index)
	
	train_df = train_df.reset_index(drop=True)
	test_df = test_df.reset_index(drop=True)
	
	train_y = train_df.loc[:,'class']
	test_y = test_df.loc[:,'class']
	
	train_X = train_df.iloc[:, 1:]
	test_X = test_df.iloc[:, 1:]
	
	return train_X, train_y, test_X, test_y

# Code for part 5
def confusion_matrix(truth, predictions):
	conf_mat = cfm(truth, predictions)
	
	return conf_mat

# Code for bonus part B
def bonusb(filename):
	pass
