# car_eval.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 06/14/16
#	
# Description    : Testing SP with the car evaluation dataset.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Testing SP with the car evaluation dataset.

Random Forest : 91.040 %
Linear SVM    : 73.988 %

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import cPickle, os, time, csv
from math import ceil
from itertools import izip

# Third party imports
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed

# Program imports
from mHTM.region import SPRegion

# Old program imports
from encoder.category import Category as CategoryEncoder
from encoder.multi import Multi as MultiEncoder

def train_score_clf(clf, tr_x, te_x, tr_y, te_y):
	"""
	Train and fit a classifier on the data.
	"""
	
	clf.fit(tr_x, tr_y)
	return clf.score(te_x, te_y)*100

def convert_data_to_int(x, y):
	"""
	Convert the provided data to integers, given a set of data with fully
	populated values.
	"""
	
	# Create the new version of X
	x_classes = []
	for i in xrange(x.shape[1]):
		x_classes.append({item:j for j, item in enumerate(set(x[:,i]))})
	new_x = np.zeros(x.shape, dtype='i')
	for i, xi in enumerate(x):
		for j, xii in enumerate(xi):
			new_x[i,j] = x_classes[j][xii]
	
	# Create the new version of y
	y_classes = {item:i for i, item in enumerate(set(y))}
	new_y = np.zeros(y.shape, dtype='i')
	for i, yi in enumerate(y):
		new_y[i] = y_classes[yi]
	
	return new_x, new_y
		
def base_learners(data_path='data.csv', seed=123456789):
	"""
	Test some classifiers on the raw data.
	"""
	
	# Params
	nsplits = 8
	pct_train = 0.8
	
	# Get data
	data = pd.read_csv(data_path)
	x = data.ix[:, :-1].as_matrix()
	y = data.ix[:, -1].as_matrix()
	x, y = convert_data_to_int(x, y)
	
	# Run random forest in parallel
	sss = StratifiedShuffleSplit(y, n_iter=nsplits, train_size=pct_train,
		random_state=seed)
	results = Parallel(n_jobs=-1)(delayed(train_score_clf)(
		RandomForestClassifier(random_state=i), x[tr], x[te], y[tr], y[te])
		for i, (tr, te) in enumerate(sss))
	print 'Random Forest: {0:.3f} %'.format(np.median(results))
	
	# Run SVM in parallel
	sss = StratifiedShuffleSplit(y, n_iter=nsplits, train_size=pct_train,
		random_state=seed)
	results = Parallel(n_jobs=-1)(delayed(train_score_clf)(
		LinearSVC(random_state=i), x[tr], x[te], y[tr], y[te])
		for i, (tr, te) in enumerate(sss))
	print 'Linear SVM: {0:.3f} %'.format(np.median(results))

def sp_one_level(base_dir, data_path='data.csv', seed=123456789):
	"""
	Test the SP.
	"""
	
	# Make a new directory
	new_dir = os.path.join(base_dir, time.strftime('%Y%m%d-%H%M%S',
		time.localtime()))
	os.makedirs(new_dir)
	
	# Params
	nsplits = 8
	pct_train = 0.8
	
	# Get data
	data = pd.read_csv(data_path)
	x = data.ix[:, :-1].as_matrix()
	y = data.ix[:, -1].as_matrix()
	x, y = convert_data_to_int(x, y)
	
	# Create the encoder
	num_bits_per_encoder = 50
	category_encoders = [
		CategoryEncoder(
			num_categories=len(set(xi)), 
			num_bits=num_bits_per_encoder
		) for xi in x.T
	]
	total_bits = num_bits_per_encoder*len(category_encoders)
	encoder = MultiEncoder(
		*category_encoders
	)
	
	# Build the config for the SP
	ncolumns = 4096
	nactive = int(ncolumns * 0.20)
	nsynapses = 25
	seg_th = 0
	sp_config = {
		'ninputs': total_bits,
		'ncolumns': ncolumns,
		'nactive': nactive,
		'global_inhibition': True,
		'trim': 1e-4,
		'disable_boost': True,
		'seed': seed,
		
		'nsynapses': nsynapses,
		'seg_th': seg_th,
		
		'syn_th': 0.5,
		'pinc': 0.001,
		'pdec': 0.001,
		'pwindow': 0.5,
		'random_permanence': True,
		
		'nepochs': 1,
		'log_dir': os.path.join(new_dir, '1-1'),
		'clf': LinearSVC(random_state=seed)
	}
	
	# Encode all of the data
	new_x = np.zeros((len(x), total_bits), dtype='bool')
	for i in xrange(len(x)):
		encoder.bind_data([(x[i,j], j) for j in xrange(x.shape[1])])
		new_x[i] = np.array(list(encoder.encode()), dtype='bool')
	
	# Dump the data and the details
	with open(os.path.join(new_dir, 'input.pkl'), 'wb') as f:
		cPickle.dump((new_x, y), f, cPickle.HIGHEST_PROTOCOL)
	with open(os.path.join(new_dir, 'details.csv'), 'wb') as f:
		writer = csv.writer(f)
		category_encoder_details = [[
			'Category {0}: Num bits: {1}'.format(i, c.num_bits),
			'Category {0}: Active bits: {1}'.format(i, c.active_bits),
			'Category {0}: Num categories: {1}'.format(i, c.num_categories)]
			for i, c in enumerate(category_encoders)]
		writer.writerows(category_encoder_details)
		writer.writerow(['Num splits', nsplits])
		writer.writerow(['% train', pct_train])
		writer.writerow(['Seed', seed])
	
	# Run the experiment
	sss = StratifiedShuffleSplit(y, n_iter=nsplits, train_size=pct_train,
		random_state=seed)
	results = Parallel(n_jobs=-1)(delayed(train_score_clf)(
		SPRegion(**sp_config), new_x[tr], new_x[te], y[tr], y[te])
		for i, (tr, te) in enumerate(sss))
	pct_accuracy = np.median(results)
	print ['{0:.3f}'.format(r) for r in results]
	print 'SP + Linear SVM: {0:.3f} %'.format(pct_accuracy)
	with open(os.path.join(new_dir, 'details.csv'), 'ab') as f:
		writer = csv.writer(f)
		writer.writerow(['% Accuracy', pct_accuracy])

if __name__ == '__main__':
	base_dir = os.path.join(os.path.expanduser('~'), 'scratch', 'car_eval')
	
	# base_learners()
	sp_one_level(base_dir)
