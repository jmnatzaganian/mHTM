# mnist_novelty_detection.py
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
Experiment for using the SP for novelty detection with MNIST.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import os, cPickle, json, sys

# Third party imports
import numpy as np
from scipy.stats import uniform, randint
from sklearn.svm import OneClassSVM

# Program imports
from mHTM.region import SPRegion
from mHTM.datasets.loader import load_mnist
from mHTM.parallel import create_runner, execute_runner, ParamGenerator
from mHTM.metrics import SPMetrics

def parallel_params(log_dir, niter=10000, seed=123456789):
	"""
	Create the parameters for a parallel run.
	
	@param log_dir: The directory to store the results in.
	
	@param niter: The number of iterations to perform.
	
	@param seed: The seed for the random number generators.
	
	@return: Returns a tuple containing the parameters.
	"""
	
	static_params = {
		'ninputs': 784,
		'trim': 1e-4,
		'disable_boost': True,
		'seed': seed,
		'pct_active': None,
		'random_permanence': True,
		'pwindow': 0.5,		
		'global_inhibition': True,
		'syn_th': 0.5,
		'pinc': 0.001,
		'pdec': 0.001,		
		'nepochs': 10
	}
	dynamic_params = {
		'ncolumns': randint(500, 3500),
		'nactive': uniform(0.5, 0.35), # As a % of the number of columns
		'nsynapses': randint(25, 784),
		'seg_th': uniform(0, 0.2), # As a % of the number of synapses
		'log_dir': log_dir
	}
	
	# Build the parameter generator
	gen = ParamGenerator(dynamic_params, niter, 1, 784)
	params = {key:gen for key in dynamic_params}
	
	return static_params, params

def static_params(log_dir, seed=123456789):
	"""
	Create the parameters for a parallel run.
	
	@param log_dir: The directory to store the results in.
	
	@param seed: The seed for the random number generators.
	
	@return: The configuration parameters.
	"""
	
	params = {
		'ninputs': 784,
		'trim': 1e-4,
		'disable_boost': True,
		'seed': seed,
		'pct_active': None,
		'random_permanence': True,
		'pwindow': 0.5,
		
		'global_inhibition': True,
		
		'ncolumns': 784,
		'nactive': 39,
		
		'nsynapses': 50,
		'seg_th': 0,
		
		'syn_th': 0.5,
		
		'pinc': 0.001,
		'pdec': 0.001,
		
		'nepochs': 10,
		
		'log_dir': log_dir
	}
	
	return params

def base_experiment(config, ntrials=1, seed=123456789):
	"""
	Run a single experiment, locally.
		
	@param config: The configuration parameters to use for the SP.
	
	@param ntrials: The number of times to repeat the experiment.
	
	@param seed: The random seed to use.
	
	@return: A tuple containing the percentage errors for the SP's training
	and testing results and the SVM's training and testing results,
	respectively.
	"""
	
	# Base parameters
	ntrain, ntest = 800, 200
	clf_th = 0.5
	
	# Seed numpy
	np.random.seed(seed)
	
	# Get the data
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	tr_x_0 = np.random.permutation(tr_x[tr_y == 0])
	x_tr = tr_x_0[:ntrain]
	x_te = tr_x_0[ntrain:ntrain + ntest]
	outliers = [np.random.permutation(tr_x[tr_y == i])[:ntest] for i in
		xrange(1, 10)]
	
	# Metrics
	metrics = SPMetrics()
	
	# Get the metrics for the datasets
	u_x_tr = metrics.compute_uniqueness(x_tr)
	o_x_tr = metrics.compute_overlap(x_tr)
	c_x_tr = 1 - metrics.compute_distance(x_tr)
	u_x_te = metrics.compute_uniqueness(x_te)
	o_x_te = metrics.compute_overlap(x_te)
	c_x_te = 1 - metrics.compute_distance(x_te)
	u_y_te, o_y_te, c_y_te = [], [], []
	for outlier in outliers:
		u_y_te.append(metrics.compute_uniqueness(outlier))
		o_y_te.append(metrics.compute_overlap(outlier))
		c_y_te.append(1 - metrics.compute_distance(outlier))
	
	# Initialize the overall results
	sp_x_results = np.zeros(ntrials)
	sp_y_results = [np.zeros(ntrials) for _ in xrange(9)]
	svm_x_results = np.zeros(ntrials)
	svm_y_results = [np.zeros(ntrials) for _ in xrange(9)]
	
	# Iterate across the trials:
	for nt in xrange(ntrials):
		# Make a new seed
		config['seed'] = np.random.randint(1000000)
		
		# Create the SP
		sp = SPRegion(**config)
		
		# Fit the SP
		sp.fit(x_tr)
		
		# Get the SP's output
		sp_x_tr = sp.predict(x_tr)
		sp_x_te = sp.predict(x_te)
		sp_y_te = [sp.predict(outlier) for outlier in outliers]
		
		# Get the metrics for the SP's results
		u_sp_x_tr = metrics.compute_uniqueness(sp_x_tr)
		o_sp_x_tr = metrics.compute_overlap(sp_x_tr)
		c_sp_x_tr = 1 - metrics.compute_distance(sp_x_tr)
		u_sp_x_te = metrics.compute_uniqueness(sp_x_te)
		o_sp_x_te = metrics.compute_overlap(sp_x_te)
		c_sp_x_te = 1 - metrics.compute_distance(sp_x_te)
		u_sp_y_te, o_sp_y_te, c_sp_y_te = [], [], []
		for y in sp_y_te:
			u_sp_y_te.append(metrics.compute_uniqueness(y))
			o_sp_y_te.append(metrics.compute_overlap(y))
			c_sp_y_te.append(1 - metrics.compute_distance(y))
		
		# Log all of the metrics
		sp._log_stats('Input Base Class Train Uniqueness', u_x_tr)
		sp._log_stats('Input Base Class Train Overlap', o_x_tr)
		sp._log_stats('Input Base Class Train Correlation', c_x_tr)
		sp._log_stats('Input Base Class Test Uniqueness', u_x_te)
		sp._log_stats('Input Base Class Test Overlap', o_x_te)
		sp._log_stats('Input Base Class Test Correlation', c_x_te)
		sp._log_stats('SP Base Class Train Uniqueness', u_sp_x_tr)
		sp._log_stats('SP Base Class Train Overlap', o_sp_x_tr)
		sp._log_stats('SP Base Class Train Correlation', c_sp_x_tr)
		sp._log_stats('SP Base Class Test Uniqueness', u_sp_x_te)
		sp._log_stats('SP Base Class Test Overlap', o_sp_x_te)
		sp._log_stats('SP Base Class Test Correlation', c_sp_x_te)
		for i, (a, b, c, d, e, f) in enumerate(zip(u_y_te, o_y_te, c_y_te,
			u_sp_y_te, o_sp_y_te, c_sp_y_te), 1):
			sp._log_stats('Input Novelty Class {0} Uniqueness'.format(i), a)
			sp._log_stats('Input Novelty Class {0} Overlap'.format(i), b)
			sp._log_stats('Input Novelty Class {0} Correlation'.format(i), c)	
			sp._log_stats('SP Novelty Class {0} Uniqueness'.format(i), d)
			sp._log_stats('SP Novelty Class {0} Overlap'.format(i), e)
			sp._log_stats('SP Novelty Class {0} Correlation'.format(i), f)
		
		# Get average representation of the base class
		sp_base_result = np.mean(sp_x_tr, 0)
		sp_base_result[sp_base_result >= 0.5] = 1
		sp_base_result[sp_base_result < 1] = 0
		
		# Averaged results for each metric type
		u_sp_base_to_x_te = 0.
		o_sp_base_to_x_te = 0.
		c_sp_base_to_x_te = 0.
		u_sp, o_sp, c_sp = np.zeros(9), np.zeros(9), np.zeros(9)
		for i, x in enumerate(sp_x_te):
			xt = np.vstack((sp_base_result, x))
			u_sp_base_to_x_te += metrics.compute_uniqueness(xt)
			o_sp_base_to_x_te += metrics.compute_overlap(xt)
			c_sp_base_to_x_te += 1 - metrics.compute_distance(xt)
			
			for j, yi in enumerate(sp_y_te):
				yt = np.vstack((sp_base_result, yi[i]))
				u_sp[j] += metrics.compute_uniqueness(yt)
				o_sp[j] += metrics.compute_overlap(yt)
				c_sp[j] += 1 - metrics.compute_distance(yt)
		u_sp_base_to_x_te /= ntest
		o_sp_base_to_x_te /= ntest
		c_sp_base_to_x_te /= ntest
		for i in xrange(9):
			u_sp[i] /= ntest
			o_sp[i] /= ntest
			c_sp[i] /= ntest
		
		# Log the results
		sp._log_stats('Base Train to Base Test Uniqueness',
			u_sp_base_to_x_te)
		sp._log_stats('Base Train to Base Test Overlap', o_sp_base_to_x_te)
		sp._log_stats('Base Train to Base Test Correlation', c_sp_base_to_x_te)
		for i, j in enumerate(xrange(1, 10)):
			sp._log_stats('Base Train to Novelty {0} Uniqueness'.format(j),
				u_sp[i])
			sp._log_stats('Base Train to Novelty {0} Overlap'.format(j),
				o_sp[i])
			sp._log_stats('Base Train to Novelty {0} Correlation'.format(j),
				c_sp[i])
		
		# Create an SVM
		clf = OneClassSVM(kernel='linear', nu=0.1, random_state=seed2)
		
		# Evaluate the SVM's performance
		clf.fit(x_tr)
		svm_x_te = len(np.where(clf.predict(x_te) == 1)[0]) / float(ntest) * \
			100
		svm_y_te = np.array([len(np.where(clf.predict(outlier) == -1)[0]) /
			float(ntest) * 100 for outlier in outliers])
		
		# Perform classification using overlap as the feature
		# -- The overlap must be above 50%
		clf_x_te = 0.
		clf_y_te = np.zeros(9)
		for i, x in enumerate(sp_x_te):
			xt = np.vstack((sp_base_result, x))
			xo = metrics.compute_overlap(xt)
			if xo >= clf_th: clf_x_te += 1
			
			for j, yi in enumerate(sp_y_te):
				yt = np.vstack((sp_base_result, yi[i]))
				yo = metrics.compute_overlap(yt)
				if yo < clf_th: clf_y_te[j] += 1
		clf_x_te = (clf_x_te / ntest) * 100
		clf_y_te = (clf_y_te / ntest) * 100
		
		# Store the results as errors
		sp_x_results[nt] = 100 - clf_x_te
		sp_y_results[nt] = 100 - clf_y_te
		svm_x_results[nt] = 100 - svm_x_te
		svm_y_results[nt] = 100 - svm_y_te
		
		# Log the results
		sp._log_stats('SP % Correct Base Class', clf_x_te)
		sp._log_stats('SVM % Correct Base Class', svm_x_te)
		for i, j in enumerate(xrange(1, 10)):
			sp._log_stats('SP % Correct Novelty Class {0}'.format(j),
				clf_y_te[i])
			sp._log_stats('SVM % Correct Novelty Class {0}'.format(j),
				svm_y_te[i])
		sp._log_stats('SP % Mean Correct Novelty Class', np.mean(clf_y_te))
		sp._log_stats('SVM % Mean Correct Novelty Class', np.mean(svm_y_te))
		sp._log_stats('SP % Adjusted Score', (np.mean(clf_y_te) * clf_x_te) /
			100)
		sp._log_stats('SVM % Adjusted Score', (np.mean(svm_y_te) * svm_x_te) /
			100)
	
	return sp_x_results, sp_y_results, svm_x_results, svm_y_results

def slurm_prep(log_dir, niter=10000, partition_name='debug',
	this_dir=os.getcwd()):
	"""
	Prep the SLRUM runs.
	
	@param log_dir: The directory to store the results in.
	
	@param niter: The number of iterations to perform.
	
	@param partition_name: The partition name of the cluster to use.
	
	@param this_dir: The full path to the directory where this file is located.
	"""
	
	# Get the configuration details
	static_config, dynamic_config = parallel_params(log_dir, niter)
	
	# Create the runs
	for i in xrange(1, niter + 1):
		# Build the initial params
		params = {k:v.rvs() for k, v in sorted(dynamic_config.items())}
		for k, v in static_config.items():
			params[k] = v
		
		# Create the base directory
		dir = params['log_dir']
		splits = os.path.basename(dir).split('-')
		dir = os.path.join(os.path.dirname(dir),
			'-'.join(s for s in splits[:-1]))
		try:
			os.makedirs(dir)
		except OSError:
			pass
		
		# Dump the params as JSON
		s = json.dumps(params, sort_keys=True, indent=4,
			separators=(',', ': ')).replace('},', '},\n')
		with open(os.path.join(dir, 'config.json'), 'wb') as f:
			f.write(s)
		
		# Create the runner
		mnist_runner_path = os.path.join(this_dir,
			'mnist_novelty_detection.py')
		command = 'python "{0}" "{1}"'.format(mnist_runner_path, dir)
		runner_path = os.path.join(dir, 'runner.sh')
		job_name = str(i)
		stdio_path = os.path.join(dir, 'stdio.txt')
		stderr_path = os.path.join(dir, 'stderr.txt')
		create_runner(command=command, runner_path=runner_path,
			job_name=job_name, partition_name=partition_name,
			stdio_path=stdio_path, stderr_path=stderr_path,
			time_limit='00-00:45:00', memory_limit=512)
		
		# Execute the runner
		# execute_runner(runner_path)

if __name__ == '__main__':
	# local = True
	local = False
	
	user_path = os.path.expanduser('~')
	
	partition_name = 'debug'
	# partition_name = 'work'
	
	niter = 10
	# niter = 10000
	
	this_dir = os.path.join(user_path, 'mHTM', 'dev')
	
	if local:
		log_dir = os.path.join(user_path, 'scratch', 'novelty_experiments',
			'mnist')
		config = static_params(log_dir)
		base_experiment(config, 1)
	else:
		if len(sys.argv) == 1:
			log_dir = os.path.join(user_path, 'results',
				'mnist_novelty_detection')
			slurm_prep(log_dir, niter, partition_name)
		else:
			with open(os.path.join(sys.argv[1], 'config.json'), 'rb') as f:
				config = json.load(f)
			base_experiment(config, 1)
