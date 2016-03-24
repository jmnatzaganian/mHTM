import os

import numpy as np
from sklearn.svm import OneClassSVM, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from joblib import Parallel, delayed

from mHTM.datasets.loader import load_mnist, MNISTCV
from mHTM.metrics import SPMetrics
from mHTM.region import SPRegion


def main():
	"""
	Use a linear SVM for multi-class classification.
	
	One vs the rest : 77.61%
	Default         : 77.61%
	One vs one      : 85.07%
	"""
	
	seed = 123456789
	np.random.seed(seed)
	ntrain, ntest = 800, 200
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	x, y = np.vstack((tr_x, te_x)), np.hstack((tr_y, te_y))
	cv = MNISTCV(tr_y, te_y, ntrain, ntest, 1, seed)

	for tr, te in cv:
		clf = OneVsRestClassifier(LinearSVC(random_state=seed), -1)
		clf.fit(x[tr], y[tr])
		print clf.score(x[te], y[te])
		
		clf = LinearSVC(random_state=seed)
		clf.fit(x[tr], y[tr])
		print clf.score(x[te], y[te])
		
		clf = OneVsOneClassifier(LinearSVC(random_state=seed), -1)
		clf.fit(x[tr], y[tr])
		print clf.score(x[te], y[te])

def main2():
	"""
	Use one class SVM for multi-class classification
	
	Accuracy = 71.45%
	"""
	
	# Initializations
	seed = 123456789
	np.random.seed(seed)
	ntrain, ntest = 800, 200
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	tr, te = [], []
	for i in xrange(10):
		tr.append(np.random.permutation(tr_x[tr_y == i])[:ntrain])
		te.append(np.random.permutation(te_x[te_y == i])[:ntest])
	
	# Train the classifiers and get their results
	clfs = []
	for i in xrange(10):
		clf = OneClassSVM(kernel='linear', nu=0.1, random_state=seed)
		clf.fit(tr[i])
		clfs.append(clf)
		
	# Test the classifiers
	te_x = np.vstack(te)
	te_y = np.hstack([np.array([i] * ntest) for i in xrange(10)])
	results = np.zeros((10, len(te_y)))
	for i in xrange(10):
		results[i] = clfs[i].decision_function(te_x).flatten() + \
			np.random.uniform(0.1, 0.2, len(te_y))
	print np.sum(np.argmax(results, 0) == te_y) / float(len(te_y))

def _main3(params, x):
	"""
	Used by main3 to do the SP training in parallel.
	
	@param params: The configuration parameters for the SP.
	
	@param x: The data to train the SP on.
	
	@return: The SP instance, as well as its predictions on the training data.
	"""
	
	clf = SPRegion(**params)
	clf.fit(x)
	y = np.mean(clf.predict(x), 0)
	y[y >= 0.5] = 1
	y[y < 1] = 0
	
	return clf, y

def _main3_2(clf, x, base_result, seed):
	"""
	Used by main3 to do the SP testing in parallel.
	
	@param clf: An instance of the classifier
	
	@param x: The data to test the SP on.
	
	@param base_result: The SP's base result.
	
	@param seed: Seed for random number generator
	
	@return: The SP's overlap results.
	"""
	
	np.random.seed(seed)
	
	metrics = SPMetrics()
	
	y = clf.predict(x)
	
	result = np.zeros(len(y))
	for i, yi in enumerate(y):
		yt = np.vstack((base_result, yi))
		result[i] = metrics.compute_overlap(yt)
	
	# Tie-breaker
	result += np.random.uniform(0.001, 0.002, len(y))
	
	return result

def main3(log_dir):
	"""
	Use one class SP for multi-class classification
	
	Accuracy = 49.8%
	"""
	
	# Initializations
	seed = 123456789
	np.random.seed(seed)
	ntrain, ntest = 800, 200
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	tr, te = [], []
	for i in xrange(10):
		tr.append(np.random.permutation(tr_x[tr_y == i])[:ntrain])
		te.append(np.random.permutation(te_x[te_y == i])[:ntest])
	params = {
		'ninputs': 784,
		'trim': 1e-4,
		'disable_boost': True,
		'seed': seed,
		'pct_active': None,
		'random_permanence': True,
		'pwindow': 0.5,
		
		'global_inhibition': True,
		
		'ncolumns': 784,
		'nactive': 78,
		
		'nsynapses': 100,
		'seg_th': 0,
		
		'syn_th': 0.5,
		
		'pinc': 0.001,
		'pdec': 0.001,
		
		'nepochs': 10,
		
		'log_dir': log_dir
	}
	metrics = SPMetrics()
	
	# Train the classifiers
	clfs = []
	base_results = []
	for clf, y in Parallel(n_jobs=-1)(delayed(_main3)(params, tr[i])
		for i in xrange(10)):
		clfs.append(clf)
		base_results.append(y)
		
	# Test the classifiers
	te_x = np.vstack(te)
	te_y = np.hstack([np.array([i] * ntest) for i in xrange(10)])
	results = np.array(Parallel(n_jobs=-1)(delayed(_main3_2)(clfs[i], te_x,
		base_results[i], seed) for i in xrange(10)))
	
	print np.sum(np.argmax(results, 0) == te_y) / float(len(te_y))

if __name__ == '__main__':
	# main()
	# main2()
	main3(os.path.join(os.path.expanduser('~'), 'scratch',
		'mnist_novelty_classification', 'r1'))
