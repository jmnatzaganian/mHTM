# novelty_detection_slurm.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 03/22/16
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
import os, cPickle, json, sys

# Third party imports
import numpy as np
from sklearn.svm import OneClassSVM

# Program imports
from mHTM.region import SPRegion
from mHTM.datasets.loader import SPDataset
from mHTM.metrics import SPMetrics
from mHTM.parallel import create_runner, execute_runner

def generate_seeds(nseeds=10, seed=123456789):
	"""
	Create some random seeds.
	
	@param nseeds: The number of seeds to create.
	
	@param seed: The seed to use to initialize this function.
	
	@return: A NumPy array containing the seeds.
	"""
	
	# Set the random state
	np.random.seed(seed)
	return [int(x) for x in np.random.random_sample(nseeds) * 1e9]

def create_base_config(log_dir,	seed=123456789):
	"""
	Create the base configuration for the experiments.
	
	@param log_dir: The directory where this run should be created.
	
	@param seed: The random seed to use.
	
	@return: A dictionary containing the base configuration for the SP.
	"""
	
	return {
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
		'nsynapses': 75,
		'seg_th': 15,
		'syn_th': 0.5,
		'pinc': 0.001,
		'pdec': 0.001,
		'nepochs': 10,
		'log_dir': log_dir
	}

def base_experiment(config, pct_noise=0.15, noverlap_bits=0, ntrials=10,
	verbose=False, seed=123456789):
	"""
	Run a single experiment, locally.
	
	@param config: The configuration parameters.
	
	@param pct_noise: The percentage of noise to add to the dataset.
	
	@param noverlap_bits: The number of bits the base class should overlap
	with the novelty class.
	
	@param ntrials: The number of times to repeat the experiment.
	
	@param verbose: If True print the results.
	
	@param seed: The random seed to use.
	"""
	
	# Base parameters
	ntrain, ntest = 800, 200
	nsamples, nbits, pct_active = ntest + ntrain, 100, 0.4
	clf_th = 0.5
	
	# Build the directory, if needed
	base_dir = config['log_dir']
	if not os.path.exists(base_dir): os.makedirs(base_dir)
	
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
	
	if verbose:
		print "\nBase class' test noise: {0:2.2f}".format(1 - (np.mean(x_te, 0)
			* x_ds.base_class.astype('i')).sum() / 40.)
		print "Outlier's class noise: {0:2.2f}".format(1 - (np.mean(y_te, 0) *
			outlier_base.astype('i')).sum() / 40.)
		print 'Overlap between two classes: {0}'.format(np.dot(
			x_ds.base_class.astype('i'), outlier_base.astype('i')))
	
	# Metrics
	metrics = SPMetrics()
	
	# Get the metrics for the datasets
	u_x_tr = metrics.compute_uniqueness(x_tr)
	o_x_tr = metrics.compute_overlap(x_tr)
	u_x_te = metrics.compute_uniqueness(x_te)
	o_x_te = metrics.compute_overlap(x_te)
	u_y_te = metrics.compute_uniqueness(y_te)
	o_y_te = metrics.compute_overlap(y_te)
	
	# Initialize the overall results
	sp_x_results = np.zeros(ntrials)
	sp_y_results = np.zeros(ntrials)
	svm_x_results = np.zeros(ntrials)
	svm_y_results = np.zeros(ntrials)
	
	# Iterate across the trials:
	for i, seed2 in enumerate(generate_seeds(ntrials, seed)):
		# Create the SP
		config['seed'] = seed2
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
		u_sp_x_te = metrics.compute_uniqueness(sp_x_te)
		o_sp_x_te = metrics.compute_overlap(sp_x_te)
		u_sp_y_te = metrics.compute_uniqueness(sp_y_te)
		o_sp_y_te = metrics.compute_overlap(sp_y_te)
		
		# Log all of the metrics
		sp._log_stats('Input Base Class Train Uniqueness', u_x_tr)
		sp._log_stats('Input Base Class Train Overlap', o_x_tr)
		sp._log_stats('Input Base Class Test Uniqueness', u_x_te)
		sp._log_stats('Input Base Class Test Overlap', o_x_te)
		sp._log_stats('Input Novelty Class Test Uniqueness', u_y_te)
		sp._log_stats('Input Novelty Class Test Overlap', o_y_te)
		sp._log_stats('SP Base Class Train Uniqueness', u_sp_x_tr)
		sp._log_stats('SP Base Class Train Overlap', o_sp_x_tr)
		sp._log_stats('SP Base Class Test Uniqueness', u_sp_x_te)
		sp._log_stats('SP Base Class Test Overlap', o_sp_x_te)
		sp._log_stats('SP Novelty Class Test Uniqueness', u_sp_y_te)
		sp._log_stats('SP Novelty Class Test Overlap', o_sp_y_te)
		
		# Print the results
		fmt_s = '{0}:\t{1:2.4f}\t{2:2.4f}\t{3:2.4f}\t{4:2.4f}\t{5:2.4f}\t{6:2.4f}'
		if verbose:
			print '\nDescription\tx_tr\tx_te\ty_te\tsp_x_tr\tsp_x_te\tsp_y_te'
			print fmt_s.format('Uniqueness', u_x_tr, u_x_te, u_y_te, u_sp_x_tr,
				u_sp_x_te, u_sp_y_te)
			print fmt_s.format('Overlap', o_x_tr, o_x_te, o_y_te, o_sp_x_tr,
				o_sp_x_te, o_sp_y_te)
		
		# Get average representation of the base class
		sp_base_result = np.mean(sp_x_tr, 0)
		sp_base_result[sp_base_result >= 0.5] = 1
		sp_base_result[sp_base_result < 1] = 0
		
		# Averaged results for each metric type
		u_sp_base_to_x_te = 0.
		o_sp_base_to_x_te = 0.
		u_sp_base_to_y_te = 0.
		o_sp_base_to_y_te = 0.
		for x, y in zip(sp_x_te, sp_y_te):
			# Refactor
			xt = np.vstack((sp_base_result, x))
			yt = np.vstack((sp_base_result, y))
			
			# Compute the sums
			u_sp_base_to_x_te += metrics.compute_uniqueness(xt)
			o_sp_base_to_x_te += metrics.compute_overlap(xt)
			u_sp_base_to_y_te += metrics.compute_uniqueness(yt)
			o_sp_base_to_y_te += metrics.compute_overlap(yt)
		u_sp_base_to_x_te /= ntest
		o_sp_base_to_x_te /= ntest
		u_sp_base_to_y_te /= ntest
		o_sp_base_to_y_te /= ntest
		
		# Log the results
		sp._log_stats('Base Train to Base Test Uniqueness',
			u_sp_base_to_x_te)
		sp._log_stats('Base Train to Base Test Overlap', o_sp_base_to_x_te)
		sp._log_stats('Base Train to Novelty Test Uniqueness',
			u_sp_base_to_y_te)
		sp._log_stats('Base Train to Novelty Test Overlap', o_sp_base_to_y_te)
		
		# Print the results
		if verbose:
			print '\nDescription\tx_tr->x_te\tx_tr->y_te'
			print 'Uniqueness:\t{0:2.4f}\t{1:2.4f}'.format(u_sp_base_to_x_te,
				u_sp_base_to_y_te)
			print 'Overlap:\t{0:2.4f}\t{1:2.4f}'.format(o_sp_base_to_x_te,
				o_sp_base_to_y_te)
		
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
	
	# Save the results
	with open(os.path.join(base_dir, 'results.pkl'), 'wb') as f:
		cPickle.dump((sp_x_results, sp_y_results, svm_x_results,
			svm_y_results), f, cPickle.HIGHEST_PROTOCOL)

def slurm_prep(log_dir, partition_name='debug', this_dir=os.getcwd()):
	"""
	Prep the SLRUM runs.
	
	@param log_dir: The directory to store the results in.
	
	@param partition_name: The partition name of the cluster to use.
	
	@param this_dir: The full path to the directory where this file is located.
	"""
	
	# Create the runs
	i = 1
	for noise in np.linspace(0, 1, 101):
		for overlap in np.arange(0, 41):
			dir = os.path.join(log_dir, '{0}-{1}'.format(noise, overlap))
			
			# Create the base directory
			try:
				os.makedirs(dir)
			except OSError:
				pass
			
			# Dump the params as JSON
			s = json.dumps(create_base_config(dir), sort_keys=True,
				indent=4, separators=(',', ': ')).replace('},', '},\n')
			with open(os.path.join(dir, 'config.json'), 'wb') as f:
				f.write(s)
			
			# Create the runner
			mnist_runner_path = os.path.join(this_dir,
				'novelty_detection_slurm.py')
			command = 'python "{0}" "{1}" "{2}" "{3}"'.format(
				mnist_runner_path, dir, noise, overlap)
			runner_path = os.path.join(dir, 'runner.sh')
			job_name = str(i)
			stdio_path = os.path.join(dir, 'stdio.txt')
			stderr_path = os.path.join(dir, 'stderr.txt')
			create_runner(command=command, runner_path=runner_path,
				job_name=job_name, partition_name=partition_name,
				stdio_path=stdio_path, stderr_path=stderr_path,
				time_limit='00-00:10:00', memory_limit=128)
			
			# Execute the runner
			execute_runner(runner_path)
			
			i += 1

if __name__ == '__main__':
	# local = True
	local = False
	
	user_path = os.path.expanduser('~')
	this_dir = os.path.join(user_path, 'mHTM', 'dev', 'novelty_detection')
	
	partition_name = 'debug'
	# partition_name = 'work'
	
	if local:
		log_dir = os.path.join(user_path, 'scratch', 'novelty_detection',
			'0.15-0')
		config = create_base_config(log_dir)
		base_experiment(config, ntrials=10, verbose=True)
	else:
		if len(sys.argv) == 1:
			log_dir = os.path.join(user_path, 'results', 'novelty_detection')
			slurm_prep(log_dir, partition_name, this_dir)
		else:
			with open(os.path.join(sys.argv[1], 'config.json'), 'rb') as f:
				config = json.load(f)
			base_experiment(config, float(sys.argv[2]), int(sys.argv[3]))
