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
import os, cPickle, time

# Third party imports
import numpy as np
from scipy.spatial.distance import pdist

# Program imports
from mHTM.region import SPRegion
from mHTM.plot import plot_line

###############################################################################
##### Classes
###############################################################################

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
	def compute_total_similarity(data, confidence_interval=0.9):
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
		data_temp = data.astype('f')
		mean = data_temp.mean(0)
		
		# Compute number of positions that are within the confidence interval
		nabove = np.sum(mean >= confidence_interval)
		nbelow = np.sum(mean <= 1 - confidence_interval)
		
		return (nabove + nbelow) / float(data_temp.shape[1])
	
	@staticmethod
	def compute_one_similarity(data, confidence_interval=0.9):
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
		data_temp = data.astype('f')
		mean = data_temp.mean(0)
		
		# Compute number of positions that are within the confidence interval
		nabove = np.sum(mean >= confidence_interval)
		nbelow = np.sum(mean <= 1 - confidence_interval)
		
		return nabove / float(data_temp.shape[1] - nbelow)
	
	@staticmethod
	def compute_zero_similarity(data, confidence_interval=0.9):
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
		data_temp = data.astype('f')
		mean = data_temp.mean(0)
		
		# Compute number of positions that are within the confidence interval
		nabove = np.sum(mean >= confidence_interval)
		nbelow = np.sum(mean <= 1 - confidence_interval)
		
		return nbelow / float(data_temp.shape[1] - nabove)
	
	@staticmethod
	def compute_dissimilarity(data, confidence_interval=0.9):
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
		data_temp = data.astype('f')
		mean = data_temp.mean(0)
		
		# Compute number of positions that are within the confidence interval
		nabove = np.sum(mean >= confidence_interval)
		nbelow = np.sum(mean <= 1 - confidence_interval)
		
		# The number of bits
		nbits = float(data_temp.shape[1])
		
		return (nbits - (nabove + nbelow)) / nbits
	
	@staticmethod
	def compute_overlap(data):
		"""
		Compute the average normalized overlap across all vector pairs. The
		overlap is normalized relative to the largest overlap possible for the
		given data.
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@return: The mean normalized overlap.
		"""
		
		# Find the "max" overlap
		x = data.astype('f')
		s = np.sum(x, 1)
		max_overlap = np.min(s[np.argsort(s)[-2:]])
		
		# Build the return object
		m, n = data.shape
		
		# Compute the average distance across all vector pairs
		s = c = 0.
		for i in xrange(0, m - 1):
			for j in xrange(i + 1, m):
				s += np.dot(x[i], x[j])
				c += 1
		
		return (s / c) / max_overlap
	
	@staticmethod	
	def compute_distance(data, metric='correlation'):
		"""
		Compute the average distance between all vector pairs using the
		provided distances. Refer to U{pdist<http://docs.scipy.org/doc/
		scipy-0.16.0/reference/generated/scipy.spatial.distance.pdist.html>}
		for supported distances.
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@param metric: The distance metric to use.
		
		@return: The mean distance.
		"""
		
		# Compute the average distance across all vector pairs
		return pdist(data.astype('f'), metric).mean()

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
	base_path = os.path.join(os.path.expanduser('~'), 'scratch', '1')
	log_dir = os.path.join(base_path, '1-1')
	seed = 123456789 # 852
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
	# uniqueness_sp, uniqueness_data = np.zeros(npoints), np.zeros(npoints)
	overlap_sp, overlap_data = np.zeros(npoints), np.zeros(npoints)
	# correlation_sp, correlation_data = np.zeros(npoints), np.zeros(npoints)
	
	# Metrics
	metrics = SPMetrics()
	
	# Vary input noise
	for i, pct_noise in enumerate(pct_noises):
		print i
		
		# Build the dataset
		ds = SPDataset(nsamples=nsamples, nbits=nbits, pct_active=pct_active,
			pct_noise=pct_noise, seed=seed)
		
		# Get the dataset stats
		# uniqueness_data[i] = metrics.compute_uniqueness(ds.data)
		overlap_data[i] = metrics.compute_overlap(ds.data)
		# correlation_data[i] = 1 - metrics.compute_distance(ds.data)
		
		# See if the data already exists
		try:
			d1, d2 = os.path.basename(log_dir).split('-')
			path = os.path.join(base_path, '{0}-{1}'.format(d1, i+1),
				'column_activations-train.pkl')
			with open(path, 'rb') as f:
				sp_output = cPickle.load(f)
		except OSError:
			# Build the SP
			sp = SPRegion(**kargs)
			
			# Train the region
			sp.fit(ds.data)
			
			# Get the SP's output SDRs
			sp_output = sp.predict(ds.data)
		
		# Get the stats
		# uniqueness_sp[i] = metrics.compute_uniqueness(sp_output)
		overlap_sp[i] = metrics.compute_overlap(sp_output)
		# correlation_sp[i] = 1 - metrics.compute_distance(sp_output)
	
	# Make some plots
	# plot_line([pct_noises * 100, pct_noises * 100], [uniqueness_data * 100,
		# uniqueness_sp * 100], series_names=('Raw Data', 'SP Output'),
		# x_label='% Noise', y_label='Uniqueness [%]', xlim=False, ylim=False,
		# out_path=os.path.join(base_path, 'uniqueness.png'), show=False)
	plot_line([pct_noises * 100, pct_noises * 100], [overlap_data * 100,
		overlap_sp * 100], series_names=('Raw Data', 'SP Output'),
		x_label='% Noise', y_label="% Normalized Overlap", xlim=False,
		ylim=False, out_path=os.path.join(base_path, 'overlap.png'),
		show=False)
	# plot_line([pct_noises * 100, pct_noises * 100], [correlation_data * 100,
		# correlation_sp * 100], series_names=('Raw Data', 'SP Output'),
		# x_label='% Noise', y_label="% Correlation", xlim=False,
		# ylim=False, out_path=os.path.join(base_path, 'correlation.png'),
		# show=False)

if __name__ == '__main__':
	main()
