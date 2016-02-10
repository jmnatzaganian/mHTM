# parameter_exploration.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 02/07/16
# 
# Description    : Experiment for studying parameter affects.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Experiment for studying parameter affects.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import json, os, pkgutil, csv, cPickle

# Third party imports
import numpy as np

# Program imports
from mHTM.region import SPRegion

###############################################################################
##### Classes
###############################################################################

class SPDataset(object):
	"""
	Class for working with a dataset specifically designed for the SP.
	
	The dataset consists of a single class. It will consist of the number of
	desired samples. Each sample is an SDR with the specified number of total
	bits and the specified percent of active bits. The actual class SDR will be
	chosen randomly. Using the desired amount of noise, bits will be randomly
	flipped to populate the dataset.
	"""
	
	def __init__(self, nsamples=500, nbits=100, pct_active=0.4, pct_noise=0.1,
		seed=None):
		"""
		Initialize the class.
		
		@param nsamples: The number of samples to add to the dataset.
		
		@param nbits: The number of bits each sample should have.
		
		@param pct_active: The percentage of bits that will be active in the
		base class SDR.
		
		@param pct_noise: The percentage of noise to add to the data.
		
		@param seed: The seed used to initialize the random number generator.
		"""
		
		# Store the parameters
		self.nsamples = nsamples
		self.nbits = nbits
		self.pct_active = pct_active
		self.pct_noise = pct_noise
		self.seed = seed
		
		# Convert percentages to integers
		nactive = int(nbits * pct_active)
		noise = int(nbits * pct_noise)
		
		# Keep a random number generator internally to ensure the global state
		# is unaltered
		self.prng = np.random.RandomState()
		self.prng.seed(self.seed)
		
		# Create the base class SDR
		self.input = np.zeros(self.nbits, dtype='bool')
		self.input[self.prng.choice(self.nbits, nactive, False)] = 1
		
		# Initialize the dataset
		self.data = np.repeat(self.input.reshape(1, self.nbits), self.nsamples,
			axis=0)
		
		# Add noise to the dataset
		for i in xrange(len(self.data)):
			sel = self.prng.choice(self.nbits, noise, False)
			self.data[i, sel] = np.bitwise_not(self.data[i, sel])
	
class SPMetrics(object):
	"""
	This class allows for an unbiased method for studying the SP. Custom
	scoring metrics are included for determining how good the SP's output SDRs
	are.
	
	The included metrics are currently only for a single class; however, it
	should be possible to expand these to support multi-class.
	"""
	
	@staticmethod
	def compute_uniqueness(data):
		"""
		Compute the percentage of unique SDRs in the given dataset. This method
		will return the percentage of unique SDRs. It is normalized such that
		if all SDRs are unique it will return one and if no SDRs are alike it
		will return zero. A score of zero indicates that exactly the same SDRs
		were produced.
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@return: The percentage of unique SDRs.
		"""
		
		nunique, nsamples = len(set([tuple(d) for d in data])), len(data)
		
		return (nunique - 1) / (nsamples - 1.)
	
	@staticmethod
	def compute_similarity(data, confidence_interval=0.9):
		"""
		Compute the degree of similarity between SDRs. This method computes
		the average activation of each bit across the SDRs. For a bit to be
		similar across all SDRs it must be active at least confidence_interval%
		of the time or it must be inactive at least (1 - confidence_interval)%
		of the time. If each bit in the SDR meets that criteria, the SDRs are
		said to be 100% similar (this method returns a 1).
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@param nactive: The expected number of active bits in each SDR.
		
		@param confidence_interval: A threshold used to determine your
		definition of similarity. Any bit in the SDR that is active at least
		this percentage of the time will be considered to be valid.
		
		@return: The percentage of similarity.
		"""
		
		# Compute the mean across rows
		mean = data.mean(0)
		
		# Compute number of positions that within the confidence interval
		nabove = np.sum(mean >= confidence_interval)
		nbelow = np.sum(mean <= 1 - confidence_interval)
		
		return (nabove + nbelow) / float(data.shape[1])

if __name__ == '__main__':
	# Create the dataset
	ds = SPDataset(seed=123456789)
	
	# Reference the metrics
	metrics = SPMetrics()
	
	print metrics.compute_uniqueness(ds.data)
	print metrics.compute_similarity(ds.data)