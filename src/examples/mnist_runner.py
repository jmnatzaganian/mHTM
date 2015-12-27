# mnist_runner.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 12/05/15
# 
# Description    : SLURM runner for MNIST.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
SLURM runner for MNIST.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import cPickle, os, json, sys, shutil

# Third party imports
import numpy as np
from sklearn.svm import LinearSVC

# Program imports
from mHTM.datasets.loader import load_mnist
from mHTM.region import SPRegion

def full_cv(base_dir):
	"""
	Run the MNIST experiment. Each CV split is executed sequentially.
	
	@param base_dir: The full path to the base directory. This directory should
	contain the config as well as the pickled data.
	"""
	
	# Get the keyword arguments for the SP
	with open(os.path.join(base_dir, 'config.json'), 'rb') as f:
		kargs = json.load(f)
	kargs['clf'] = LinearSVC(random_state=kargs['seed'])
	
	# Get the data
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	x, y = np.vstack((tr_x, te_x)), np.hstack((tr_y, te_y))
	
	# Get the CV splits
	with open(os.path.join(base_dir, 'cv.pkl'), 'rb') as f:
		cv = cPickle.load(f)
	
	# Execute each run
	for tr, te in cv:
		clf = SPRegion(**kargs)
		clf.fit(x[tr], y[tr])
		
		# Column accuracy
		clf.score(x[te], y[te])
		
		# Probabilistic accuracy
		clf.score(x[te], y[te], tr_x=x[tr], score_method='prob')
		
		# Dimensionality reduction method
		clf.score(x[te], y[te], tr_x=x[tr], score_method='reduction')
		ndims = len(clf.reduce_dimensions(x[0]))
		clf._log_stats('Number of New Dimensions', ndims)

def one_cv(base_dir, cv_split):
	"""
	Run the MNIST experiment. Only the specified CV split is executed.
	
	@param base_dir: The full path to the base directory. This directory should
	contain the config as well as the pickled data.
	
	@param cv_split: The index for the CV split.
	"""
	
	# Get the keyword arguments for the SP
	with open(os.path.join(base_dir, 'config-{0}.json'.format(cv_split)),
		'rb') as f:
		kargs = json.load(f)
	kargs['clf'] = LinearSVC(random_state=kargs['seed'])
	
	# Get the data
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	x, y = np.vstack((tr_x, te_x)), np.hstack((tr_y, te_y))
	
	# Get the CV splits
	with open(os.path.join(base_dir, 'cv.pkl'), 'rb') as f:
		cv = cPickle.load(f)
	tr, te = cv[cv_split - 1]
	
	# Remove the split directory, if it exists
	shutil.rmtree(os.path.join(base_dir, str(cv_split)), True)
	
	# Execute
	clf = SPRegion(**kargs)
	clf.fit(x[tr], y[tr])
	
	# Column accuracy
	clf.score(x[te], y[te])
	
	# Probabilistic accuracy
	clf.score(x[te], y[te], tr_x=x[tr], score_method='prob')
	
	# Dimensionality reduction method
	clf.score(x[te], y[te], tr_x=x[tr], score_method='reduction')
	ndims = len(clf.reduce_dimensions(x[0]))
	clf._log_stats('Number of New Dimensions', ndims)

def full_mnist(base_dir, new_dir, auto_update=False):
	"""
	Execute a full MNIST run using the parameters specified by ix.
	
	@param base_dir: The full path to the base directory. This directory should
	contain the config.
	
	@param new_dir: The full path of where the data should be saved.
	
	@param auto_update: If True the permanence increment and decrement amounts
	will automatically be computed by the runner. If False, the ones specified
	in the config file will be used.
	"""
	
	# Get the keyword arguments for the SP
	with open(os.path.join(base_dir, 'config.json'), 'rb') as f:
		kargs = json.load(f)
	kargs['log_dir'] = new_dir
	kargs['clf'] = LinearSVC(random_state=kargs['seed'])
	
	# Get the data
	(tr_x, tr_y), (te_x, te_y) = load_mnist()

	# Manually compute the permanence update amounts
	if auto_update:
		# Compute average sum of each training instance
		avg_s = tr_x.sum(1)
		
		# Compute the total average sum
		avg_ts = avg_s.mean()
		
		# Compute the average active probability
		a_p = avg_ts / float(tr_x.shape[1])
		
		# Compute the scaling factor
		scaling_factor = 1 / avg_ts
		
		# Compute the update amounts
		pinc = scaling_factor * (1 / a_p)
		pdec = scaling_factor * (1 / (1 - a_p))
		
		# Update the config
		kargs['pinc'], kargs['pdec'] = pinc, pdec
	
	# Execute
	clf = SPRegion(**kargs)
	clf.fit(tr_x, tr_y)
	
	# Column accuracy
	clf.score(te_x, te_y)
	
	# Probabilistic accuracy
	clf.score(te_x, te_y, tr_x=tr_x, score_method='prob')
	
	# Dimensionality reduction method
	clf.score(te_x, te_y, tr_x=tr_x, score_method='reduction')
	ndims = len(clf.reduce_dimensions(tr_x[0]))
	clf._log_stats('Number of New Dimensions', ndims)

if __name__ == '__main__':
	if len(sys.argv) == 2:
		full_cv(sys.argv[1])
	elif len(sys.argv) == 3:
		try:
			one_cv(sys.argv[1], int(sys.argv[2]))
		except ValueError: # Value was a string
			full_mnist(sys.argv[1], sys.argv[2])
	elif len(sys.argv) == 4:
		full_mnist(sys.argv[1], sys.argv[2], bool(int(sys.argv[3])))