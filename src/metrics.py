# metrics.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 02/20/16
#	
# Description    : Module for computing various metrics
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Module for computing various metrics.

G{packagetree mHTM}
"""

# Third-Party imports
import numpy as np
from scipy.spatial.distance import pdist

###############################################################################
# Quality of output metrics
###############################################################################

class SPMetrics(object):
	"""
	This class allows for an unbiased method for studying the SP. The items in
	this class are specifically designed for characterizing the quality of
	SDRs. Custom scoring metrics are included for determining how good the SP's
	output SDRs are.
	
	The included metrics are currently only for a single class. In other words,
	the data you pass to one of these methods must all belong to the same
	class. For evaluating datasets with multiple classes, each class should be
	evaluating independently. Averaging or a similar metric could be used to
	obtain an overall metric.
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
		Compute the degree of '1' similarity between SDRs. This method computes
		the average activation of each bit across the SDRs. For a bit to be
		similar across all SDRs it must be active at least confidence_interval%
		of the time. This method only looks at the similarity of the active
		bits. If those bits in the SDRs meet the above criteria, the SDRs are
		said to be 100% similar (this method returns a 1).
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@param nactive: The expected number of active bits in each SDR.
		
		@param confidence_interval: A threshold used to determine your
		definition of similarity. Any bit in the SDR that is active at least
		this percentage of the time will be considered to be valid.
		
		@return: The percentage of one similarity.
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
		Compute the degree of '0' similarity between SDRs. This method computes
		the average activation of each bit across the SDRs. For a bit to be
		similar across all SDRs it must be inactive at least
		(1 - confidence_interval)% of the time. This method only looks at the
		similarity of the inactive bits. If those bits in the SDRs meet the
		above criteria, the SDRs are said to be 100% similar
		(this method returns a 1).
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@param nactive: The expected number of active bits in each SDR.
		
		@param confidence_interval: A threshold used to determine your
		definition of similarity. Any bit in the SDR that is active at least
		this percentage of the time will be considered to be valid.
		
		@return: The percentage of zero similarity.
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
		Compute the degree of dissimilarity between SDRs. This method is used
		to report the percentage of bits that do not fall into an active or
		inactive category (see the compute_one_similarity and
		compute_zero_similarity for more details). If all bits are similar,
		this method returns a 0, i.e. they are 0% dissimilar.
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@param nactive: The expected number of active bits in each SDR.
		
		@param confidence_interval: A threshold used to determine your
		definition of similarity. Any bit in the SDR that is active at least
		this percentage of the time will be considered to be valid.
		
		@return: The percentage of dissimilarity.
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
