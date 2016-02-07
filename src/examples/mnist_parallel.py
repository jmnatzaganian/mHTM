# mnist_parallel.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 12/02/15
# 
# Description    : Testing SP with MNIST using parameter optimization and doing
# iterations in parallel.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Testing SP with MNIST using parameter optimization and doing iterations in
parallel.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import cPickle, os, json, pkgutil

# Third party imports
import numpy as np
from scipy.stats import uniform, randint
from sklearn.grid_search import RandomizedSearchCV
from sklearn.svm import LinearSVC

# Program imports
from mHTM.region import SPRegion
from mHTM.parallel import create_runner, execute_runner, ParamGenerator
from mHTM.datasets.loader import load_mnist, MNISTCV

def main(ntrain=800, ntest=200, niter=10, nsplits=5, global_inhibition=True,
	seed=None):
	"""
	Build the information needed to perform CV on a subset of the MNIST
	dataset.
	
	@param ntrain: The number of training samples to use.
	
	@param ntest: The number of testing samples to use.
	
	@param niter: The number of parameter iterations to use.
	
	@param nsplits: The number of splits of the data to use.
	
	@param global_inhibition: If True use global inhibition; otherwise, use
	local inhibition.
	
	@param seed: The seed for the random number generators.
	
	@return: The full set of X, the full set of Y, the keyword arguments for
	the classifier, the params for CV, and the CV.
	"""
	
	# Get the data
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	x, y = np.vstack((tr_x, te_x)), np.hstack((tr_y, te_y))
	cv = MNISTCV(tr_y, te_y, ntrain, ntest, nsplits, seed)
			
	# Create static parameters
	ninputs = tr_x.shape[1]
	kargs = {
		# Region parameters
		'ninputs': ninputs,
		'global_inhibition': global_inhibition,
		'trim': 1e-4,
		'seed': seed,
		
		# Synapse parameters
		'syn_th': 0.5,
		'random_permanence': True,
		
		# Fitting parameters
		'nepochs': 30,
		'clf': LinearSVC(random_state=seed)
		# NOTE: The SVM's will be identical, despite being seeded now
	}
	
	# Come up with some parameters to search
	param_distributions = {
		# Region parameters
		'ncolumns':randint(100, 1001),
		'nactive':uniform(0, 0.2),
		# As a percentage of the number of columns
		
		# Column parameters
		'max_boost': randint(1, 21),
		'duty_cycle': randint(10, 1001),
		
		# Segment parameters
		'nsynapses': randint(1, ninputs + 1),
		'seg_th': uniform(0, 0.1),
		# As a percentage of the number of synapses
		
		# Synapse parameters
		'pinc': uniform(0.001, 0.1),
		'pdec': uniform(0.001, 0.1),
		'pwindow': uniform(0.001, 0.1),
		
		# Fitting parameters
		'log_dir': log_dir
	}
	
	# Build the parameter generator
	gen = ParamGenerator(param_distributions, niter, nsplits, ninputs)
	params = {key:gen for key in param_distributions}
	
	return x, y, kargs, params, cv

def main_local(log_dir, ntrain=800, ntest=200, niter=5, nsplits=3,
	global_inhibition=True, ncores=4, seed=None):
	"""
	Perform CV on a subset of the MNIST dataset. Performs parallelizations on
	a local machine.
	
	@param log_dir: The directory to store the results in.
	
	@param ntrain: The number of training samples to use.
	
	@param ntest: The number of testing samples to use.
	
	@param niter: The number of parameter iterations to use.
	
	@param nsplits: The number of splits of the data to use.
	
	@param global_inhibition: If True use global inhibition; otherwise, use
	local inhibition.
	
	@param ncores: The number of cores to use.
	
	@param seed: The seed for the random number generators.
	"""
	
	# Run the initialization
	x, y, kargs, params, cv = main(ntrain, ntest, niter, nsplits, seed)
	
	# Build the classifier for doing CV
	clf = RandomizedSearchCV(
		estimator=SPRegion(**kargs),
		param_distributions=params,
		n_iter=niter, # Total runs
		n_jobs=ncores, # Use this many number of cores
		pre_dispatch=2 * ncores, # Give each core two jobs at a time
		iid=True, # Data is iid across folds
		cv=cv, # The CV split for the data
		refit=False, # Disable fitting best estimator on full dataset
		random_state=seed # Force same SP across runs
	)
	
	# Fit the models
	clf.fit(x, y)
	
	# Extract the CV results
	parameter_names = sorted(clf.grid_scores_[0].parameters.keys())
	parameter_names.pop(parameter_names.index('log_dir'))
	parameter_values = np.zeros((niter, len(parameter_names)))
	results = np.zeros((niter, nsplits))
	for i, score in enumerate(clf.grid_scores_):
		parameter_values[i] = np.array([score.parameters[k] for k in
			parameter_names])
		results[i] = score.cv_validation_scores
	
	# Save the CV results
	with open(os.path.join(log_dir, 'cv_results.pkl'), 'wb') as f:
		cPickle.dump((parameter_names, parameter_values, results), f,
			cPickle.HIGHEST_PROTOCOL)
	with open(os.path.join(log_dir, 'cv_clf.pkl'), 'wb') as f:
		cPickle.dump((clf.grid_scores_, clf.best_score_, clf.best_params_), f,
			cPickle.HIGHEST_PROTOCOL)

def main_slurm(log_dir, ntrain=800, ntest=200, niter=5, nsplits=3,
	global_inhibition=True, partition_name='debug', seed=None):
	"""
	Perform CV on a subset of the MNIST dataset, using SLRUM. Iterations will
	be run in complete parallel. Splits within an iteration will be run
	sequentially.
	
	@param log_dir: The directory to store the results in.
	
	@param ntrain: The number of training samples to use.
	
	@param ntest: The number of testing samples to use.
	
	@param niter: The number of parameter iterations to use.
	
	@param nsplits: The number of splits of the data to use.
	
	@param global_inhibition: If True use global inhibition; otherwise, use
	local inhibition.
	
	@param partition_name: The partition name of the cluster to use.
	
	@param seed: The seed for the random number generators.
	"""
	
	# Run the initialization
	x, y, kargs, params, cv = main(ntrain, ntest, niter, nsplits,
		global_inhibition, seed)
	
	# Create the runs
	for i in xrange(1, niter + 1):
		# Build the initial params
		param = {k:v.rvs() for k, v in sorted(params.items())}
		
		# Create the base directory
		dir = param['log_dir']
		splits = os.path.basename(dir).split('-')
		dir = os.path.join(os.path.dirname(dir),
			'-'.join(s for s in splits[:-1]))
		try:
			os.makedirs(dir)
		except OSError:
			pass
		
		# Dump the CV data
		with open(os.path.join(dir, 'cv.pkl'), 'wb') as f:
			cPickle.dump(list(cv), f, cPickle.HIGHEST_PROTOCOL)
		
		# Build the full params
		for k, v in kargs.items():
			if k != 'clf': # Add the classifier later
				param[k] = v
		
		# Dump the params as JSON
		s = json.dumps(param, sort_keys=True, indent=4,
			separators=(',', ': ')).replace('},', '},\n')
		with open(os.path.join(dir, 'config.json'), 'wb') as f:
			f.write(s)
		
		# Create the runner
		mnist_runner_path = os.path.join(pkgutil.get_loader('mHTM.examples').
			filename, 'mnist_runner.py')
		command = 'python "{0}" "{1}"'.format(mnist_runner_path, dir)
		runner_path = os.path.join(dir, 'runner.sh')
		job_name = str(i)
		stdio_path = os.path.join(dir, 'stdio.txt')
		stderr_path = os.path.join(dir, 'stderr.txt')
		create_runner(command=command, runner_path=runner_path,
			job_name=job_name, partition_name=partition_name,
			stdio_path=stdio_path, stderr_path=stderr_path)
		
		# Execute the runner
		execute_runner(runner_path)

if __name__ == '__main__':
	ntrain, ntest, niter, nsplits, ncores = 800, 200, 1000, 5, 32
	global_inhibition, partition_name, seed = True, 'work', 123456789
	log_dir = 'results/partial_mnist-global'
	np.random.seed(seed) # To ensure consistency
	
	# Run on local
	# main_local(log_dir, ntrain, ntest, niter, nsplits, global_inhibition,
		# ncores, seed)
	
	# Run on cluster
	global_inhibition = True
	log_dir = 'results/partial_mnist-global'
	main_slurm(log_dir, ntrain, ntest, niter, nsplits, global_inhibition,
		partition_name, seed=seed)
	global_inhibition = False
	log_dir = 'results/partial_mnist-local'
	main_slurm(log_dir, ntrain, ntest, niter, nsplits, global_inhibition,
		partition_name, seed=seed)