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
import os, cPickle, json

# Third party imports
import numpy as np
from sklearn.svm import LinearSVC

# Program imports
from mHTM.datasets.loader import load_mnist
from mHTM.region import SPRegion

def main():
	base_dir = r'C:\Users\james\scratch\test\995'
	with open(os.path.join(base_dir, 'config.json'), 'rb') as f:
		kargs = json.load(f)
	kargs['clf'] = LinearSVC(random_state=kargs['seed'])
	kargs['log_dir'] = os.path.join(os.path.dirname(base_dir), 'local')
	
	# Get the data
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	x, y = np.vstack((tr_x, te_x)), np.hstack((tr_y, te_y))
	
	# Get the CV splits
	with open(os.path.join(base_dir, 'cv.pkl'), 'rb') as f:
		cv = cPickle.load(f)
	tr, te = cv[0]
	
	clf = SPRegion(**kargs)
	clf.fit(x[tr], y[tr])
	clf.score(x[te], y[te])
	clf.score(x[te], y[te], tr_x=x[tr], score_method='prob')
	clf.score(x[te], y[te], tr_x=x[tr], score_method='reduction')
	ndims = len(clf.reduce_dimensions(x[0]))
	clf._log_stats('Number of New Dimensions', ndims)

def main2():
	base_dir = r'C:\Users\james\scratch\test\local'
	file = 'column_activations-test'
	
	with open(os.path.join(base_dir, '{0}.pkl'.format(file)), 'rb') as f:
		local = cPickle.load(f)
	with open(os.path.join(base_dir, '{0}_2.pkl'.format(file)), 'rb') as f:
		remote = cPickle.load(f)
	
	print np.all(local == remote)
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	# main()
	main2()