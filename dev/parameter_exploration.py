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
import os

# Third party imports
import numpy as np

# Program imports
from mHTM.region import SPRegion
from mHTM.plot import plot_line

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

###############################################################################
##### Functions
###############################################################################

def main():
	"""
	Program entry.
	
	Build an SP using SPDataset and see how it performs.
	"""
	
	# Params
	nsamples, nbits, pct_active = 500, 100, 0.4
	ncolumns = 300
	base_path = os.path.join(os.path.expanduser('~'), 'scratch')
	log_dir = os.path.join(base_path, '1-1')
	seed = 123456789
	kargs = {
		'ninputs': nbits,
		'ncolumns': 300,
		'nactive': 0.02 * ncolumns,
		'global_inhibition': True,
		'trim': 1e-4,
		'disable_boost': True,
		'seed': seed,
		
		'nsynapses': 20,
		'seg_th': 2,
		
		'syn_th': 0.5,
		'pinc': 0.01,
		'pdec': 0.01,
		'pwindow': 0.5,
		'random_permanence': True,
		
		'nepochs': 1,
		'log_dir': log_dir
	}
	
	# Build items to store results
	npoints = 100
	pct_noises = np.linspace(0, pct_active / 2, npoints, False)
	uniqueness_sp, uniqueness_data = np.zeros(npoints), np.zeros(npoints)
	similarity_sp, similarity_data = np.zeros(npoints), np.zeros(npoints)
	
	# Metrics
	metrics = SPMetrics()
	
	# Vary input noise
	for i, pct_noise in enumerate(pct_noises):
		# Build the dataset
		ds = SPDataset(nsamples=nsamples, nbits=nbits, pct_active=pct_active,
			pct_noise=pct_noise, seed=seed)
		
		# Get the dataset stats
		uniqueness_data[i] = metrics.compute_uniqueness(ds.data)
		similarity_data[i] = metrics.compute_similarity(ds.data,
			confidence_interval=0.9)
		
		# Build the SP
		sp = SPRegion(**kargs)
		
		# Train the region
		sp.fit(ds.data)
		
		# Get the SP's output SDRs
		sp_output = sp.predict(ds.data)
		
		# Get the stats
		uniqueness_sp[i] = metrics.compute_uniqueness(sp_output)
		similarity_sp[i] = metrics.compute_similarity(sp_output,
			confidence_interval=0.9)
	
	# Make some plots
	plot_line([pct_noises * 100, pct_noises * 100], [uniqueness_data * 100,
		uniqueness_sp * 100], series_names=('Raw Data', 'SP Output'),
		x_label='% Noise', y_label='Uniqueness [%]', xlim=False, ylim=False,
		out_path=os.path.join(base_path, 'uniqueness.png'))
	plot_line([pct_noises * 100, pct_noises * 100], [similarity_data * 100,
		similarity_sp * 100], series_names=('Raw Data', 'SP Output'),
		x_label='% Noise', y_label='Similarity [%]', xlim=False, ylim=False,
		out_path=os.path.join(base_path, 'similarity.png'))

if __name__ == '__main__':
	main()
	
	# import cPickle
	# with open(r'C:\Users\james\scratch\1-100\column_activations-train.pkl') as f:
		# data = cPickle.load(f)
	
	# import pdb; pdb.set_trace()