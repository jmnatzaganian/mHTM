# mnist_simple.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 12/13/15
#	
# Description    : Testing SP with MNIST using a simple demonstration.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

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

def main(ntrain=800, ntest=200, nsplits=1, seed=1234567):
	# Set the configuration parameters for the SP
	ninputs = 784
	kargs = {
		'ninputs': ninputs,
		'ncolumns': ninputs,
		'nactive': 10,
		'global_inhibition': True,
		'trim': False,
		'seed': seed,
		
		'disable_boost': True,
		
		'nsynapses': 392,
		'seg_th': 10,
		
		'syn_th': 0.5,
		'pinc': 0.001,
		'pdec': 0.002,
		'pwindow': 0.01,
		'random_permanence': True,
		
		'nepochs': 10,
		'clf': LinearSVC(random_state=seed),
		'log_dir': os.path.join('simple_mnist', '1-1')
	}
	
	# Seed numpy
	np.random.seed(seed)
	
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
	
	# Get a random set of unique inputs from the training set
	inputs = np.zeros((10, ninputs))
	for i in xrange(10):
		ix = np.random.permutation(np.where(y[tr] == i)[0])[0]
		inputs[i] = x[tr][ix]
	
	# Get the SP's predictions for the inputs
	sp_pred = sp.predict(inputs)
	
	# Get the reconstruction in the context of the SP
	sp_inputs = sp.reconstruct_input(sp_pred)
	
	# Make a plot comparing the images
	shape = (28, 28)
	path = os.path.join(sp.log_dir, 'input_reconstruction.png')
	plot_compare_images((inputs, sp_pred, sp_inputs), shape, out_path=path)

if __name__ == '__main__':
	main()