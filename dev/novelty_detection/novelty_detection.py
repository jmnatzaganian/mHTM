# novelty_detection.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 03/13/16
# 
# Description    : Experiment for using the SP for novelty detection.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Experiment for using the SP for novelty detection.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import os, cPickle

# Third party imports
import numpy as np
from sklearn.svm import OneClassSVM
from joblib import Parallel, delayed

# Program imports
from mHTM.region import SPRegion
from mHTM.datasets.loader import SPDataset
from mHTM.metrics import SPMetrics
from mHTM.plot import plot_error, compute_err

def base_experiment(pct_noise=0.15, noverlap_bits=0, exp_name='1-1',
	ntrials=10, verbose=True, seed=123456789):
	"""
	Run a single experiment, locally.
	
	@param pct_noise: The percentage of noise to add to the dataset.
	
	@param noverlap_bits: The number of bits the base class should overlap
	with the novelty class.
	
	@param exp_name: The name of the experiment.
	
	@param ntrials: The number of times to repeat the experiment.
	
	@param verbose: If True print the results.
	
	@param seed: The random seed to use.
	
	@return: A tuple containing the percentage errors for the SP's training
	and testing results and the SVM's training and testing results,
	respectively.
	"""
	
	# Base parameters
	ntrain, ntest = 800, 200
	nsamples, nbits, pct_active = ntest + ntrain, 100, 0.4
	clf_th = 0.5
	log_dir = os.path.join(os.path.expanduser('~'), 'scratch',
		'novelty_experiments', exp_name)
	
	# Configure the SP
	config = {
		'ninputs': 100,
		'trim': 1e-4,
		'disable_boost': True,
		'seed': seed,
		'pct_active': None,
		'random_permanence': True,
		'pwindow': 0.5,
		
		'global_inhibition': True,
		
		'ncolumns': 200,
		'nactive': 50,
		
		
		'nsynapses': 100,
		'seg_th': 5,
		
		'syn_th': 0.5,
		
		'pinc': 0.001,
		'pdec': 0.001,
		
		'nepochs': 10,
		
		'log_dir': log_dir
	}
	
	# Seed numpy
	np.random.seed(seed)
	
	# Create the base dataset
	x_ds = SPDataset(nsamples, nbits, pct_active, pct_noise, seed=seed)
	x_tr, x_te = x_ds.data[:ntrain], x_ds.data[ntrain:]
	
	# Create the outlier dataset
	base_indexes = set(np.where(x_ds.base_class == 1)[0])
	choices = [x for x in xrange(nbits) if x not in base_indexes]
	outlier_base = np.zeros(nbits, dtype='bool')
	outlier_base[np.random.choice(choices, x_ds.nactive - noverlap_bits,
		False)] = 1
	outlier_base[np.random.permutation(list(base_indexes))[:noverlap_bits]] = 1
	y_ds = SPDataset(ntest, nbits, pct_active, pct_noise, outlier_base, seed)
	y_te = y_ds.data
	
	import pdb; pdb.set_trace()
	
	# Metrics
	metrics = SPMetrics()
	
	# Get the metrics for the datasets
	u_x_tr = metrics.compute_uniqueness(x_tr)
	o_x_tr = metrics.compute_overlap(x_tr)
	c_x_tr = 1 - metrics.compute_distance(x_tr)
	u_x_te = metrics.compute_uniqueness(x_te)
	o_x_te = metrics.compute_overlap(x_te)
	c_x_te = 1 - metrics.compute_distance(x_te)
	u_y_te = metrics.compute_uniqueness(y_te)
	o_y_te = metrics.compute_overlap(y_te)
	c_y_te = 1 - metrics.compute_distance(y_te)
	
	# Initialize the overall results
	sp_x_results = np.zeros(ntrials)
	sp_y_results = np.zeros(ntrials)
	svm_x_results = np.zeros(ntrials)
	svm_y_results = np.zeros(ntrials)
	
	# Iterate across the trials:
	for i in xrange(ntrials):
		# Make a new seed
		seed2 = np.random.randint(1000000)
		config['seed'] = seed2
		config['log_dir'] = '{0}-{1}'.format(log_dir, i + 1)
		
		# Create the SP
		sp = SPRegion(**config)
		
		# Fit the SP
		sp.fit(x_tr)
		
		# Get the SP's output
		sp_x_tr = sp.predict(x_tr)
		sp_x_te = sp.predict(x_te)
		sp_y_te = sp.predict(y_te)
		
		# Get the metrics for the SP's results
		u_sp_x_tr = metrics.compute_uniqueness(sp_x_tr)
		o_sp_x_tr = metrics.compute_overlap(sp_x_tr)
		c_sp_x_tr = 1 - metrics.compute_distance(sp_x_tr)
		u_sp_x_te = metrics.compute_uniqueness(sp_x_te)
		o_sp_x_te = metrics.compute_overlap(sp_x_te)
		c_sp_x_te = 1 - metrics.compute_distance(sp_x_te)
		u_sp_y_te = metrics.compute_uniqueness(sp_y_te)
		o_sp_y_te = metrics.compute_overlap(sp_y_te)
		c_sp_y_te = 1 - metrics.compute_distance(sp_y_te)
		
		# Log all of the metrics
		sp._log_stats('Input Base Class Train Uniqueness', u_x_tr)
		sp._log_stats('Input Base Class Train Overlap', o_x_tr)
		sp._log_stats('Input Base Class Train Correlation', c_x_tr)
		sp._log_stats('Input Base Class Test Uniqueness', u_x_te)
		sp._log_stats('Input Base Class Test Overlap', o_x_te)
		sp._log_stats('Input Base Class Test Correlation', c_x_te)
		sp._log_stats('Input Novelty Class Test Uniqueness', u_y_te)
		sp._log_stats('Input Novelty Class Test Overlap', o_y_te)
		sp._log_stats('Input Novelty Class Test Correlation', c_y_te)	
		sp._log_stats('SP Base Class Train Uniqueness', u_sp_x_tr)
		sp._log_stats('SP Base Class Train Overlap', o_sp_x_tr)
		sp._log_stats('SP Base Class Train Correlation', c_sp_x_tr)
		sp._log_stats('SP Base Class Test Uniqueness', u_sp_x_te)
		sp._log_stats('SP Base Class Test Overlap', o_sp_x_te)
		sp._log_stats('SP Base Class Test Correlation', c_sp_x_te)
		sp._log_stats('SP Novelty Class Test Uniqueness', u_sp_y_te)
		sp._log_stats('SP Novelty Class Test Overlap', o_sp_y_te)
		sp._log_stats('SP Novelty Class Test Correlation', c_sp_y_te)
		
		# Print the results
		fmt_s = '{0}:\t{1:2.4f}\t{2:2.4f}\t{3:2.4f}\t{4:2.4f}\t{5:2.4f}\t{5:2.4f}'
		if verbose:
			print 'Description\tx_tr\tx_te\ty_te\tsp_x_tr\tsp_x_te\tsp_y_te'
			print fmt_s.format('Uniqueness', u_x_tr, u_x_te, u_y_te, u_sp_x_tr,
				u_sp_x_te, u_sp_y_te)
			print fmt_s.format('Overlap', o_x_tr, o_x_te, o_y_te, o_sp_x_tr, o_sp_x_te,
				o_sp_y_te)
			print fmt_s.format('Correlation', c_x_tr, c_x_te, c_y_te, c_sp_x_tr,
				c_sp_x_te, c_sp_y_te)
		
		# Get average representation of the base class
		sp_base_result = np.mean(sp_x_tr, 0)
		sp_base_result[sp_base_result >= 0.5] = 1
		sp_base_result[sp_base_result < 1] = 0
		
		# Averaged results for each metric type
		u_sp_base_to_x_te = 0.
		o_sp_base_to_x_te = 0.
		c_sp_base_to_x_te = 0.
		u_sp_base_to_y_te = 0.
		o_sp_base_to_y_te = 0.
		c_sp_base_to_y_te = 0.
		for x, y in zip(sp_x_te, sp_y_te):
			# Refactor
			xt = np.vstack((sp_base_result, x))
			yt = np.vstack((sp_base_result, y))
			
			# Compute the sums
			u_sp_base_to_x_te += metrics.compute_uniqueness(xt)
			o_sp_base_to_x_te += metrics.compute_overlap(xt)
			c_sp_base_to_x_te += 1 - metrics.compute_distance(xt)
			u_sp_base_to_y_te += metrics.compute_uniqueness(yt)
			o_sp_base_to_y_te += metrics.compute_overlap(yt)
			c_sp_base_to_y_te += 1 - metrics.compute_distance(yt)
		u_sp_base_to_x_te /= ntest
		o_sp_base_to_x_te /= ntest
		c_sp_base_to_x_te /= ntest
		u_sp_base_to_y_te /= ntest
		o_sp_base_to_y_te /= ntest
		c_sp_base_to_y_te /= ntest
		
		# Log the results
		sp._log_stats('Base Train to Base Test Uniqueness',
			u_sp_base_to_x_te)
		sp._log_stats('Base Train to Base Test Overlap', o_sp_base_to_x_te)
		sp._log_stats('Base Train to Base Test Correlation', c_sp_base_to_x_te)
		sp._log_stats('Base Train to Novelty Test Uniqueness',
			u_sp_base_to_y_te)
		sp._log_stats('Base Train to Novelty Test Overlap', o_sp_base_to_y_te)
		sp._log_stats('Base Train to Novelty Test Correlation',
			c_sp_base_to_y_te)
		
		# Print the results
		if verbose:
			print '\nDescription\tx_tr->x_te\tx_tr->y_te'
			print 'Uniqueness:\t{0:2.4f}\t{1:2.4f}'.format(u_sp_base_to_x_te,
				u_sp_base_to_y_te)
			print 'Overlap:\t{0:2.4f}\t{1:2.4f}'.format(o_sp_base_to_x_te,
				o_sp_base_to_y_te)
			print 'Correlation:\t{0:2.4f}\t{1:2.4f}'.format(c_sp_base_to_x_te,
				c_sp_base_to_y_te)
		
		# Create an SVM
		clf = OneClassSVM(kernel='linear', nu=0.1, random_state=seed2)
		
		# Evaluate the SVM's performance
		clf.fit(x_tr)
		svm_x_te = len(np.where(clf.predict(x_te) == 1)[0]) / float(ntest) * \
			100
		svm_y_te = len(np.where(clf.predict(y_te) == -1)[0]) / float(ntest) * \
			100
		
		# Perform classification using overlap as the feature
		# -- The overlap must be above 50%
		clf_x_te = 0.
		clf_y_te = 0.
		for x, y in zip(sp_x_te, sp_y_te):
			# Refactor
			xt = np.vstack((sp_base_result, x))
			yt = np.vstack((sp_base_result, y))
			
			# Compute the accuracy
			xo = metrics.compute_overlap(xt)
			yo = metrics.compute_overlap(yt)
			if xo >= clf_th: clf_x_te += 1
			if yo < clf_th: clf_y_te += 1
		clf_x_te = (clf_x_te / ntest) * 100
		clf_y_te = (clf_y_te / ntest) * 100
		
		# Store the results as errors
		sp_x_results[i] = 100 - clf_x_te
		sp_y_results[i] = 100 - clf_y_te
		svm_x_results[i] = 100 - svm_x_te
		svm_y_results[i] = 100 - svm_y_te
		
		# Log the results
		sp._log_stats('SP % Correct Base Class', clf_x_te)
		sp._log_stats('SP % Correct Novelty Class', clf_y_te)
		sp._log_stats('SVM % Correct Base Class', svm_x_te)
		sp._log_stats('SVM % Correct Novelty Class', svm_y_te)
		
		# Print the results
		if verbose:
			print '\nSP Base Class Detection     : {0:2.2f}%'.format(clf_x_te)
			print 'SP Novelty Class Detection  : {0:2.2f}%'.format(clf_y_te)
			print 'SVM Base Class Detection    : {0:2.2f}%'.format(svm_x_te)
			print 'SVM Novelty Class Detection : {0:2.2f}%'.format(svm_y_te)
	
	return sp_x_results, sp_y_results, svm_x_results, svm_y_results

def main(ntrials=10, seed=123456798):
	"""
	Vary the amount of noise and overlap and see how the system performs.
	
	@param ntrials: The number of trials to perform.
	
	@param seed: The seed for the random number generator.
	"""
	
	# Amount to vary each type by
	pct_noises = np.linspace(0, 1, 101)
	noverlap_bits = np.arange(0, 41)
	
	# Vary the noise
	# results = Parallel(n_jobs=-1)(delayed(base_experiment)(noise, 0,
		# 'noise-{0}'.format(i), ntrials, False, seed) for i, noise in enumerate(
		# pct_noises, 1))
	# noise_x, noise_y, svm_noise_x, svm_noise_y = [], [], [], []
	# noise_x_err1, noise_x_err2, noise_y_err1, noise_y_err2 = [], [], [], []
	# svm_noise_x_err1, svm_noise_x_err2 = [], []
	# svm_noise_y_err1, svm_noise_y_err2 = [], []
	# for sp_x, sp_y, svm_x, svm_y in results:
		# noise_x.append(np.median(sp_x))
		# noise_y.append(np.median(sp_y))
		# svm_noise_x.append(np.median(svm_x))
		# svm_noise_y.append(np.median(svm_y))
		# e = compute_err(sp_x, axis=None)
		# noise_x_err1.append(e[0])
		# noise_x_err2.append(e[1])
		# e = compute_err(sp_y, axis=None)
		# noise_y_err1.append(e[0])
		# noise_y_err2.append(e[1])
		# e = compute_err(svm_x, axis=None)
		# svm_noise_x_err1.append(e[0])
		# svm_noise_x_err2.append(e[1])
		# e = compute_err(svm_y, axis=None)
		# svm_noise_y_err1.append(e[0])
		# svm_noise_y_err2.append(e[1])
	# noise_x_err = (svm_noise_x_err1, svm_noise_x_err2)
	# noise_y_err = (svm_noise_y_err1, svm_noise_y_err2)
	# svm_noise_x_err = (svm_noise_x_err1, svm_noise_x_err2)
	# svm_noise_y_err = (svm_noise_y_err1, svm_noise_y_err2)
	
	# Vary the overlaps
	results = Parallel(n_jobs=-1)(delayed(base_experiment)(0.15, overlap,
		'overlap-{0}'.format(i), ntrials, False, seed) for i, overlap in
		enumerate(noverlap_bits, 1))
	overlap_x, overlap_y, svm_overlap_x, svm_overlap_y = [], [], [], []
	overlap_x_err1, overlap_x_err2 = [], []
	overlap_y_err1, overlap_y_err2 = [], []
	svm_overlap_x_err1, svm_overlap_x_err2 = [], []
	svm_overlap_y_err1, svm_overlap_y_err2 = [], []
	for sp_x, sp_y, svm_x, svm_y in results:
		overlap_x.append(np.median(sp_x))
		overlap_y.append(np.median(sp_y))
		svm_overlap_x.append(np.median(svm_x))
		svm_overlap_y.append(np.median(svm_y))
		e = compute_err(sp_x, axis=None)
		overlap_x_err1.append(e[0])
		overlap_x_err2.append(e[1])
		e = compute_err(sp_y, axis=None)
		overlap_y_err1.append(e[0])
		overlap_y_err2.append(e[1])
		e = compute_err(svm_x, axis=None)
		svm_overlap_x_err1.append(e[0])
		svm_overlap_x_err2.append(e[1])
		e = compute_err(svm_y, axis=None)
		svm_overlap_y_err1.append(e[0])
		svm_overlap_y_err2.append(e[1])
	overlap_x_err = (svm_overlap_x_err1, svm_overlap_x_err2)
	overlap_y_err = (svm_overlap_y_err1, svm_overlap_y_err2)
	svm_overlap_x_err = (svm_overlap_x_err1, svm_overlap_x_err2)
	svm_overlap_y_err = (svm_overlap_y_err1, svm_overlap_y_err2)
	
	# # Save the results
	p = os.path.join(os.path.expanduser('~'), 'scratch', 'novelty_experiments')
	with open(os.path.join(p, 'results.pkl'), 'wb') as f:
		# cPickle.dump((noise_x, noise_y), f, cPickle.HIGHEST_PROTOCOL)
		# cPickle.dump((svm_noise_x, svm_noise_y), f, cPickle.HIGHEST_PROTOCOL)
		# cPickle.dump((noise_x_err, noise_y_err), f, cPickle.HIGHEST_PROTOCOL)
		# cPickle.dump((svm_noise_x_err, svm_noise_y_err), f,
			# cPickle.HIGHEST_PROTOCOL)
		cPickle.dump((overlap_x, overlap_y), f, cPickle.HIGHEST_PROTOCOL)
		cPickle.dump((svm_overlap_x, svm_overlap_y), f,
			cPickle.HIGHEST_PROTOCOL)		
		cPickle.dump((overlap_x_err, overlap_y_err), f,
			cPickle.HIGHEST_PROTOCOL)
		cPickle.dump((svm_overlap_x_err, svm_overlap_y_err), f,
			cPickle.HIGHEST_PROTOCOL)
	
	# Make the plots
	# plot_error((pct_noises * 100, pct_noises * 100), (noise_x, svm_noise_x),
		# ('SP', 'SVM'), (noise_x_err, svm_noise_x_err), '% Noise', '% Error',
		# 'Noise: Base Class', out_path=os.path.join(p, 'noise_base.png'),
		# xlim=(-5, 105), ylim=(-5, 105), show=False)
	# plot_error((pct_noises * 100, pct_noises * 100), (noise_y, svm_noise_y),
		# ('SP', 'SVM'), (noise_y_err, svm_noise_y_err), '% Noise', '% Error',
		# 'Noise: Novelty Class', out_path=os.path.join(p, 'noise_novelty.png'),
		# xlim=(-5, 105), ylim=(-5, 105), show=False)
	noverlap_pct = noverlap_bits / 40. * 100
	plot_error((noverlap_pct, noverlap_pct), (overlap_x, svm_overlap_x),
		('SP', 'SVM'), (overlap_x_err, svm_overlap_x_err),
		'% Overlapping Bits', '% Error', 'Overlap: Base Class',
		out_path=os.path.join(p, 'overlap_base.png'), xlim=(-5, 105),
		ylim=(-5, 105),show=False)
	plot_error((noverlap_pct, noverlap_pct), (overlap_y, svm_overlap_y),
		('SP', 'SVM'), (overlap_y_err, svm_overlap_y_err),
		'% Overlapping Bits', '% Error', 'Overlap: Novelty Class',
		out_path=os.path.join(p, 'overlap_novelty.png'), xlim=(-5, 105),
		ylim=(-5, 105),show=False)

if __name__ == '__main__':
	base_experiment(ntrials=1, noverlap_bits=25)
	# main()
