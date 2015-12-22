# mnist_simple.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 12/13/15
#	
# Description    : Testing SP with MNIST using a simple demonstration.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Testing SP with MNIST using a simple demonstration.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import os

# Third party imports
import numpy as np
from sklearn.svm import LinearSVC

# Program imports
from mHTM.datasets.loader import load_mnist, MNISTCV
from mHTM.region import SPRegion
from mHTM.plot import plot_compare_images

def main(ntrain=800, ntest=200, nsplits=1, seed=123456789):
	# Set the configuration parameters for the SP
	ninputs = 784
	kargs = {
		'ninputs': ninputs,
		'ncolumns': ninputs,
		'nactive': 20,
		'global_inhibition': True,
		'trim': False,
		'seed': seed,
		
		'max_boost': 3,
		'duty_cycle': 8,
		
		'nsynapses': 392,
		'seg_th': 2,
		
		'syn_th': 0.5,
		'pinc': 0.01,
		'pdec': 0.02,
		'pwindow': 0.5,
		'random_permanence': True,
		
		'nepochs': 1,
		'clf': LinearSVC(random_state=seed),
		'log_dir': os.path.join('simple_mnist', '1-1')
	}
	
	# Get the data
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	x, y = np.vstack((tr_x, te_x)), np.hstack((tr_y, te_y))
	
	# Split the data for CV
	cv = MNISTCV(tr_y, te_y, ntrain, ntest, nsplits, seed)
	
	# Execute the SP on each fold. Additionally, get results for each fitting
	# method.
	for i, (tr, te) in enumerate(cv):
		# Create the region
		sp = SPRegion(**kargs)
		
		# Train the region
		sp.fit(x[tr], y[tr])
		
		# Test the base classifier
		clf = LinearSVC(random_state=seed)
		clf.fit(x[tr], y[tr])
		score = clf.score(x[te], y[te])
		print 'SVM Only Accuracy: {0:.2f}%'.format(score * 100)
		
		# Test the region for the column method
		score = sp.score(x[te], y[te])
		print 'Column Accuracy: {0:.2f}%'.format(score * 100)
		
		# Test the region for the probabilistic method
		score = sp.score(x[te], y[te], tr_x=x[tr], score_method='prob')
		print 'Probabilistic Accuracy: {0:.2f}%'.format(score * 100)
		
		# Test the region for the dimensionality reduction method
		score = sp.score(x[te], y[te], tr_x=x[tr], score_method='reduction')
		ndims = len(sp.reduce_dimensions(x[0]))
		print 'Input Reduced from {0} to {1}: {2:.1f}X reduction'.format(
			ninputs, ndims, ninputs / float(ndims))
		print 'Reduction Accuracy: {0:.2f}%'.format(score * 100)
	
	# Get a random set of unique inputs from the training set
	inputs = np.zeros((10, ninputs))
	for i in xrange(10):
		ix = np.random.permutation(np.where(y[tr] == i)[0])[0]
		inputs[i] = x[tr][ix]
	
	# Get the SP's predictions for the inputs
	sp_pred = sp.predict(inputs)
	
	# Get the reconstruction in the context of the SP
	sp_inputs = sp.reconstruct_input(sp_pred)
	
	# Make a plot comparing the two
	x1_labels = [str(i) for i in xrange(10)]
	x2_labels = [str(i) for i in xrange(10)]
	title = 'Input Reconstruction: Original (top), SP (bottom)'
	shape = (28, 28)
	path = os.path.join(sp.log_dir, 'input_reconstruction.png')
	plot_compare_images((inputs, sp_inputs), shape, title, (x1_labels,
		x2_labels,), path)

if __name__ == '__main__':
	main()