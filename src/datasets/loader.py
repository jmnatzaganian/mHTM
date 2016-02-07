# loader.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 12/03/15
#	
# Description    : Module for loading datasets.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Module for loading datasets.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import os, pkgutil, cPickle
from itertools import izip

# Third-Party imports
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

###############################################################################
# MNIST dataset
###############################################################################

class MNISTCV(object):
	"""
	Use CV on MNIST. A stratified shuffle split is used for the division of
	the data.
	"""
	
	def __init__(self, tr_y, te_y, train_size=800, test_size=200, nsplits=5,
		seed=None):
		"""
		Initialize this class.
		
		@param tr_y: The train labels.
		
		@param te_y: The test labels.
		
		@param train_size: The size of the train set, as a percentage or integer.
		
		@param test_size: The size of the test set, as a percentage or integer.
		
		@param nsplits: The number of splits of the data to create.
		
		@param seed: The random see to use.
		"""
		
		# Store the parameters
		self.tr_y = tr_y
		self.te_y = te_y
		self.train_size = train_size
		self.test_size = test_size
		self.nsplits = nsplits
		self.seed = seed
		
		# Create the internal generator
		self.gen = self._create_generator()
		
	def __iter__(self):
		"""
		Allow this class to be iterable. Each call yields a tuple containing
		the current training and testing split.
		"""
		
		for train, test in self._create_generator():
			yield train, test
	
	def __len__(self):
		"""
		Allow the generator's length to be obtained. Only valid upon the first
		call.
		
		@return: The length of the initial items in the generator.
		"""
		
		return self.nsplits
	
	def _create_generator(self):
		"""
		Create a generator for the data. Yield a tuple containing the current
		training and testing split.
		"""
		
		# Create the CV iterators
		sss_tr = StratifiedShuffleSplit(self.tr_y, self.nsplits,
			train_size=self.train_size, random_state=self.seed)
		sss_te = StratifiedShuffleSplit(self.te_y, self.nsplits,
			train_size=self.test_size, random_state=self.seed)
		
		# Yield each item
		for tr, te in izip(sss_tr, sss_te):
			yield tr[0], te[0] + len(self.tr_y) # Offset testing indexes

def load_mnist(threshold=255/2):
	"""
	Get the MNIST data.
	
	@param threshold: The position to threshold the data at. Values >= to the
	threshold are set to 1, all other values are set to 0. Setting this to None
	returns the raw data.
	
	@return: A tuple of tuple containing: (train_x, train_y), (test_x, test_y). 
	"""
	
	# Get the raw data
	p = os.path.join(pkgutil.get_loader('mHTM.datasets').filename, 'mnist.pkl')
	with open(p, 'rb') as f:
		(tr_x, tr_y), (te_x, te_y) = cPickle.load(f)
	
	# Threshold the data
	if threshold is not None:
		tr_x[tr_x < threshold] = 0
		tr_x[tr_x > 0] = 1
		tr_x = np.array(tr_x, dtype='bool')
		
		te_x[te_x < threshold] = 0
		te_x[te_x > 0] = 1
		te_x = np.array(te_x, dtype='bool')
	
	return (tr_x, tr_y), (te_x, te_y)
