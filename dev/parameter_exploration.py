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
import os, cPickle
from itertools import izip, combinations

# Third party imports
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist

# Program imports
from mHTM.region import SPRegion
from mHTM.plot import plot_line
from mHTM.datasets.loader import load_mnist, MNISTCV

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
	def compute_tanimoto_distance(data):
		"""
		Compute the average Tanimoto distance. A distance of '1' means the
		values coincide and a distance of '0' means they are as far apart as
		possible.
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@return: The distance.
		"""
		
		def distance(x, y):
			"""
			Compute the Tanimoto distance between x and y.
			
			@param x: A NumPy vector.
			
			@param y: A NumPy vector.
			"""
			
			dp = float(np.dot(x, y))
			return dp / (np.sum(x) + np.sum(y) - dp)
		
		# Compute the average distance across all vector pairs
		data_temp = data.astype('f')
		s = c = 0.
		for ix0, ix1 in combinations(xrange(len(data_temp)), 2):
			s += distance(data_temp[ix0], data_temp[ix1])
			c += 1
		return s / c
	
	@staticmethod
	def compute_population_kurtosis(data):
		"""
		Compute the population kurtosis.
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@return: The kurtosis.
		"""
		
		s = 0.
		ncols = data.shape[1]
		for d in scale(data.astype('f'), axis=1, with_mean=True,
			with_std=True):
			mean, std = np.mean(d), np.std(d)
			s += (np.sum(((d - mean) / std) ** 4) / float(ncols)) - 3
		return s / len(data)
	
	@staticmethod
	def compute_lifetime_kurtosis(data):
		"""
		Compute the lifetime kurtosis.
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@return: The kurtosis.
		"""
		
		s = 0.
		ncols = data.shape[1]
		for d in scale(data.T.astype('f'), axis=1, with_mean=True,
			with_std=True):
			mean, std = np.mean(d), np.std(d)
			s += (np.sum(((d - mean) / std) ** 4) / float(ncols)) - 3
		return s / len(data)
	
	@staticmethod
	def compute_overlap(data):
		"""
		Compute the average overlap.
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@return: The overlap.
		"""
		
		# Find the "max" overlap
		data_temp = data.astype('f')
		s = np.sum(data_temp, 1)
		max_overlap = np.min(s[np.argsort(s)[-2:]])
		
		# Compute the average distance across all vector pairs
		s = c = 0.
		for ix0, ix1 in combinations(xrange(len(data_temp)), 2):
			s += np.dot(data_temp[ix0], data_temp[ix1]) / max_overlap
			c += 1
		return s / c
	
	@staticmethod	
	def compute_centered_cosine_similarity(data):
		"""
		Compute the average centered cosine similarity.
		
		@param data: A NumPy array containing the data to compute. This must be
		a 2D object.
		
		@return: The centered cosine similarity.
		"""
		
		# Compute the average distance across all vector pairs
		return 1 - pdist(data.astype('f'), 'correlation').mean()

###############################################################################
##### Functions
###############################################################################

def dump_data(data, path):
	with open(path, 'wb') as f:
		cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

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
	# similarity_sp, similarity_data = np.zeros(npoints), np.zeros(npoints)
	# similarity_sp1, similarity_data1 = np.zeros(npoints), np.zeros(npoints)
	# similarity_sp0, similarity_data0 = np.zeros(npoints), np.zeros(npoints)
	# dissimilarity_sp, dissimilarity_data = np.zeros(npoints), np.zeros(npoints)
	# tanimoto_sp, tanimoto_data = np.zeros(npoints), np.zeros(npoints)
	# pkurtosis_sp, pkurtosis_data = np.zeros(npoints), np.zeros(npoints)
	# lkurtosis_sp, lkurtosis_data = np.zeros(npoints), np.zeros(npoints)
	# overlap_sp, overlap_data = np.zeros(npoints), np.zeros(npoints)
	cosine_sp, cosine_data = np.zeros(npoints), np.zeros(npoints)
	
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
		# similarity_data[i] = metrics.compute_total_similarity(ds.data,
			# confidence_interval=0.9)
		# similarity_data1[i] = metrics.compute_one_similarity(ds.data,
			# confidence_interval=0.9)
		# similarity_data0[i] = metrics.compute_zero_similarity(ds.data,
			# confidence_interval=0.9)
		# dissimilarity_data[i] = metrics.compute_dissimilarity(ds.data,
			# confidence_interval=0.9)
		# tanimoto_data[i] = metrics.compute_tanimoto_distance(ds.data)
		# pkurtosis_data[i] = metrics.compute_population_kurtosis(ds.data)
		# lkurtosis_data[i] = metrics.compute_lifetime_kurtosis(ds.data)
		# overlap_data[i] = metrics.compute_overlap(ds.data)
		cosine_data[i] = metrics.compute_centered_cosine_similarity(ds.data)
		
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
		# similarity_sp[i] = metrics.compute_total_similarity(sp_output,
			# confidence_interval=0.9)
		# similarity_sp1[i] = metrics.compute_one_similarity(sp_output,
			# confidence_interval=0.9)
		# similarity_sp0[i] = metrics.compute_zero_similarity(sp_output,
			# confidence_interval=0.9)
		# dissimilarity_sp[i] = metrics.compute_dissimilarity(sp_output,
			# confidence_interval=0.9)
		# tanimoto_sp[i] = metrics.compute_tanimoto_distance(sp_output)
		# pkurtosis_sp[i] = metrics.compute_population_kurtosis(sp_output)
		# # lkurtosis_sp[i] = metrics.compute_population_kurtosis(sp_output)
		# overlap_sp[i] = metrics.compute_overlap(sp_output)
		cosine_sp[i] = metrics.compute_centered_cosine_similarity(sp_output)
	
	# Make some plots
	# plot_line([pct_noises * 100, pct_noises * 100], [uniqueness_data * 100,
		# uniqueness_sp * 100], series_names=('Raw Data', 'SP Output'),
		# x_label='% Noise', y_label='Uniqueness [%]', xlim=False, ylim=False,
		# out_path=os.path.join(base_path, 'uniqueness.png'), show=False)
	# plot_line([pct_noises * 100, pct_noises * 100], [similarity_data * 100,
		# similarity_sp * 100], series_names=('Raw Data', 'SP Output'),
		# x_label='% Noise', y_label='Total similarity [%]', xlim=False, ylim=False,
		# out_path=os.path.join(base_path, 'similarity.png'), show=False)
	# plot_line([pct_noises * 100, pct_noises * 100], [similarity_data1 * 100,
		# similarity_sp1 * 100], series_names=('Raw Data', 'SP Output'),
		# x_label='% Noise', y_label="Similarity of '1's [%]", xlim=False,
		# ylim=False, out_path=os.path.join(base_path, 'one_similarity.png'),
		# show=False)
	# plot_line([pct_noises * 100, pct_noises * 100], [similarity_data0 * 100,
		# similarity_sp0 * 100], series_names=('Raw Data', 'SP Output'),
		# x_label='% Noise', y_label="Similarity of '0's [%]", xlim=False,
		# ylim=False, out_path=os.path.join(base_path, 'zero_similarity.png'),
		# show=False)
	# plot_line([pct_noises * 100, pct_noises * 100], [dissimilarity_data * 100,
		# dissimilarity_sp * 100], series_names=('Raw Data', 'SP Output'),
		# x_label='% Noise', y_label="Dissimilarity [%]", xlim=False,
		# ylim=False, out_path=os.path.join(base_path, 'dissimilarity.png'),
		# show=False)
	# plot_line([pct_noises * 100, pct_noises * 100], [tanimoto_data * 100,
		# tanimoto_sp * 100], series_names=('Raw Data', 'SP Output'),
		# x_label='% Noise', y_label="Tanimoto Distance [%]", xlim=False,
		# ylim=False, out_path=os.path.join(base_path, 'tanimoto.png'),
		# show=False)
	# plot_line([pct_noises * 100, pct_noises * 100], [pkurtosis_data, 
		# pkurtosis_sp], series_names=('Raw Data', 'SP Output'),
		# x_label='% Noise', y_label="Population Kurtosis", xlim=False,
		# ylim=False, out_path=os.path.join(base_path,
		# 'population_kurtosis.png'), show=False)
	# plot_line([pct_noises * 100, pct_noises * 100], [lkurtosis_data, 
		# lkurtosis_sp], series_names=('Raw Data', 'SP Output'),
		# x_label='% Noise', y_label="Lifetime Kurtosis", xlim=False,
		# ylim=False, out_path=os.path.join(base_path,
		# 'lifetime_kurtosis.png'), show=False)
	# plot_line([pct_noises * 100, pct_noises * 100], [overlap_data * 100,
		# overlap_sp * 100], series_names=('Raw Data', 'SP Output'),
		# x_label='% Noise', y_label="% Normalized Overlap", xlim=False,
		# ylim=False, out_path=os.path.join(base_path, 'overlap.png'),
		# show=False)
	plot_line([pct_noises * 100, pct_noises * 100], [cosine_data * 100,
		cosine_sp * 100], series_names=('Raw Data', 'SP Output'),
		x_label='% Noise', y_label="% Centered\nCosine Similarity", xlim=False,
		ylim=False, out_path=os.path.join(base_path, 'centered_cosine.png'),
		show=False)

def main2():
	seed, seed2 = 123456789, 852
	
	# Directories
	base_dir = os.path.join(os.path.expanduser('~'), 'scratch')
	base_path = os.path.join(base_dir, 'clf')
	
	# Noise (x-axis)
	npoints = 100
	pct_noises = np.linspace(0, 0.4 / 2, npoints, False)
	
	# Results
	y_sp, y_raw = np.zeros(npoints), np.zeros(npoints)
	
	# Get the results
	for i, pct_noise in enumerate(pct_noises):
		# Get class data
		with open(os.path.join(base_dir, '1', '1-{0}'.format(i + 1),
			'column_activations-train.pkl'), 'rb') as f:
			d1 = cPickle.load(f).astype('i')
			l1 = np.ones(len(d1)).astype('i')
		with open(os.path.join(base_dir, '2', '2-{0}'.format(i + 1),
			'column_activations-train.pkl'), 'rb') as f:
			d2 = cPickle.load(f).astype('i')
			l2 = np.zeros(len(d2)).astype('i')
			l2.fill(2)
		ds = SPDataset(nsamples=500, nbits=100, pct_active=0.4,
			pct_noise=pct_noise, seed=seed)
		ds2 = SPDataset(nsamples=500, nbits=100, pct_active=0.4,
			pct_noise=pct_noise, seed=seed2)
		data_raw = np.vstack((ds.data, ds2.data))
		data_sp = np.vstack((d1, d2))
		labels = np.hstack((l1, l2))
		sss = StratifiedShuffleSplit(labels, 1, train_size=0.8,
			random_state=seed)
		
		# Train SVM
		for tr, te in sss:
			clf = LinearSVC(random_state=seed)
			clf.fit(data_sp[tr], labels[tr])
			y_sp[i] = (1 - clf.score(data_sp[te], labels[te])) * 100
			
			clf = LinearSVC(random_state=seed)
			clf.fit(data_raw[tr], labels[tr])
			y_raw[i] = (1 - clf.score(data_raw[te], labels[te])) * 100
			break
	
	# Plot the results
	plot_line([pct_noises * 100, pct_noises * 100], [y_raw, y_sp],
		series_names=('Raw Data', 'SP Output'), x_label='% Noise',
		y_label='% Error', xlim=False, ylim=False, out_path=os.path.join(
		base_path, 'pct_error.png'))

def main3():
	# Set the configuration parameters for the SP
	ninputs = 784
	ncols = 936
	nsynapses = 353
	base_path = os.path.join(os.path.expanduser('~'), 'scratch', 'mnist')
	seed = 123456789
	kargs = {
		'ninputs': ninputs,
		'ncolumns': ncols,
		'nactive': 182,
		'global_inhibition': True,
		'trim': 1e-4,
		'disable_boost': True,
		'seed': seed,
		
		'nsynapses': nsynapses,
		'seg_th': 14,
		
		'syn_th': 0.5,
		'pinc': 0.0355,
		'pdec': 0.0024,
		'pwindow': 0.0105,
		'random_permanence': True,
		
		'nepochs': 1
	}
	
	# Get MNIST data
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	x, y = np.vstack((tr_x, te_x)), np.hstack((tr_y, te_y))
	
	# Split the data for CV
	ntrain, ntest, nsplits = 800, 200, 10
	cv = MNISTCV(tr_y, te_y, ntrain, ntest, nsplits, seed)
	
	# Storage for results
	sp_train = []
	sp_test = []
	train_labels = [[] for _ in xrange(nsplits)]
	test_labels = [[] for _ in xrange(nsplits)]
	sp_results = np.zeros(nsplits)
	svm_results = np.zeros(nsplits)
	
	# Execute SP on each fold
	for i, (tr, te) in enumerate(cv):		
		# Test each SP independently
		for j in xrange(10):
			# Build datasets
			tr_x, te_x = x[y[tr] == j], x[y[te] == j]
			
			# Update labels
			train_labels[i].extend([j] * len(tr_x))
			test_labels[i].extend([j] * len(te_x))
			
			# Create the SP
			kargs['log_dir'] = os.path.join(base_path, '{0}-1'.format(j))
			sp = SPRegion(**kargs)
			
			# Train the region
			sp.fit(tr_x)
			
			# Get SP results
			sp_tr = sp.predict(tr_x)
			sp_te = sp.predict(te_x)
			try:
				sp_train[i] = np.vstack((sp_train[i], sp_tr))
				sp_test[i] = np.vstack((sp_test[i], sp_te))
			except IndexError:
				sp_train.append(sp_tr)
				sp_test.append(sp_te)
		
		# Get the SP's result
		clf = LinearSVC(random_state=seed)
		clf.fit(sp_train[i], train_labels[i])
		sp_results[i] = (1 - clf.score(sp_test[i], test_labels[i])) * 100
		print clf.score(sp_test[i], test_labels[i])
		
		# Get the SVM's result
		clf = LinearSVC(random_state=seed)
		clf.fit(x[tr], y[tr])
		svm_results[i] = (1 - clf.score(x[te], y[te])) * 100
	
	# Print result
	print 'Average SVM % Error: {0:2.3f}%'.format(svm_results.mean())
	print 'Average SP % Error: {0:2.3f}%'.format(sp_results.mean())
	
	# Store results for future usage
	dump_data(sp_train, os.path.join(base_path, 'sp_train.pkl'))
	dump_data(sp_test, os.path.join(base_path, 'sp_test.pkl'))
	dump_data(train_labels, os.path.join(base_path, 'train_labels.pkl'))
	dump_data(test_labels, os.path.join(base_path, 'test_labels.pkl'))
	dump_data(sp_results, os.path.join(base_path, 'sp_results.pkl'))
	dump_data(svm_results, os.path.join(base_path, 'svm_results.pkl'))

def main4():
	# Set the configuration parameters for the SP
	ninputs = 784
	ncols = 936
	nsynapses = 353
	base_path = os.path.join(os.path.expanduser('~'), 'scratch', 'mnist-full')
	seed = 123456789
	kargs = {
		'ninputs': ninputs,
		'ncolumns': ncols,
		'nactive': 182,
		'global_inhibition': True,
		'trim': 1e-4,
		'disable_boost': True,
		'seed': seed,
		
		'nsynapses': nsynapses,
		'seg_th': 14,
		
		'syn_th': 0.5,
		'pinc': 0.0355,
		'pdec': 0.0024,
		'pwindow': 0.0105,
		'random_permanence': True,
		
		'nepochs': 1
	}
	
	# Get MNIST data
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	
	# Shrink dataset for now
	tr_x = tr_x[:100]
	tr_y = tr_y[:100]
	te_x = te_x[:100]
	te_y = te_y[:100]
	
	# Final labels
	sp_tr_y, sp_te_y = [], []
	
	# Test each SP independently
	for i in xrange(10):
		# Build datasets
		tr_xi, te_xi = tr_x[tr_y == i], te_x[te_y == i]
		
		# Create the SP
		kargs['log_dir'] = os.path.join(base_path, '{0}-1'.format(i))
		sp = SPRegion(**kargs)
		
		# Train the region
		sp.fit(tr_xi)
		
		# Get SP results
		tr = sp.predict(tr_xi)
		te = sp.predict(te_xi)
		
		# Store SP results
		try:
			sp_tr_x = np.vstack((sp_tr_x, tr))
			sp_te_x = np.vstack((sp_te_x, te))
		except NameError:
			sp_tr_x = tr
			sp_te_x = te
		
		# Store labels
		sp_tr_y.extend([i] * len(tr_xi))
		sp_te_y.extend([i] * len(te_xi))
	
	# Get the SP's result
	clf = LinearSVC(random_state=seed)
	clf.fit(sp_tr_x, sp_tr_y)
	sp_results = (1 - clf.score(sp_te_x, sp_te_y)) * 100
	
	# Get the SVM's result
	clf = LinearSVC(random_state=seed)
	clf.fit(tr_x, tr_y)
	svm_results = (1 - clf.score(te_x, te_y)) * 100
	
	# Print result
	print 'Average SVM % Error: {0:2.3f}%'.format(svm_results)
	print 'Average SP % Error: {0:2.3f}%'.format(sp_results)
	
	# Store results for future usage
	dump_data(sp_tr_x, os.path.join(base_path, 'sp_train.pkl'))
	dump_data(sp_te_x, os.path.join(base_path, 'sp_test.pkl'))
	dump_data(sp_tr_y, os.path.join(base_path, 'train_labels.pkl'))
	dump_data(sp_te_y, os.path.join(base_path, 'test_labels.pkl'))
	dump_data(sp_results, os.path.join(base_path, 'sp_results.pkl'))
	dump_data(svm_results, os.path.join(base_path, 'svm_results.pkl'))

def main5():
	# Set the configuration parameters for the SP
	ninputs = 784
	ncols = ninputs * 4
	nsynapses = ninputs / 3
	seg_th = int(nsynapses * 0.05)
	base_path = os.path.join(os.path.expanduser('~'), 'scratch', 'mnist-full')
	seed = 123456789
	kargs = {
		'ninputs': ninputs,
		'ncolumns': ncols,
		'nactive': int(0.1 * ncols),
		'global_inhibition': True,
		'trim': 1e-4,
		'disable_boost': True,
		'seed': seed,
		
		'nsynapses': nsynapses,
		'seg_th': seg_th,
		
		'syn_th': 0.5,
		'pinc': 0.01,
		'pdec': 0.01,
		'pwindow': 0.5,
		'random_permanence': True,
		
		'nepochs': 1
	}
	
	# Get MNIST data
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	
	# Shrink dataset for now
	# ntrain, ntest = 800, 200
	# tr_x = tr_x[:ntrain]
	# tr_y = tr_y[:ntrain]
	# te_x = te_x[:ntest]
	# te_y = te_y[:ntest]
	
	# Store results
	sp_tr_y = []
	poolers = []
	
	# Train each SP independently
	for i in xrange(10):
		# Build datasets
		tr_xi = tr_x[tr_y == i]
		
		# Create the SP
		kargs['log_dir'] = os.path.join(base_path, '{0}-1'.format(i))
		poolers.append(SPRegion(**kargs))
		sp = poolers[i]
		
		# Train the region
		sp.fit(tr_xi)
		
		# Get SP training results
		sp_tr = sp.predict(tr_xi)
		try:
			sp_tr_x = np.vstack((sp_tr_x, sp_tr))
		except NameError:
			sp_tr_x = sp_tr
		sp_tr_y.extend([i] * len(tr_xi))
	
	# Create classifier
	clf = LinearSVC(random_state=seed)
	clf.fit(sp_tr_x, sp_tr_y)
	
	# Get SP test results
	correct = 0.
	for x, y in izip(te_x, te_y):
		confidence_scores = np.zeros(10)
		confidence_classes = np.zeros(10).astype('i')
		for i, sp in enumerate(poolers):
			x2 = sp.predict(x)
			confidences = clf.decision_function(x2.reshape(1, -1))[0]
			confidence_classes[i] = np.argmax(confidences)
			confidence_scores[i] = confidences[confidence_classes[i]]
		if confidence_classes[np.argmax(confidence_scores)] == y: correct += 1
	print correct / len(te_y)

if __name__ == '__main__':
	main()
	# main2()
	# main3()
	# main4()
	# main5()