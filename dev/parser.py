# parser.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 12/08/15
# 
# Description    : Module for parsing the results.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Module for parsing the results.

####
# Cluster parsing
####

<base_directory>
	<param_iteration>
		config.json  # SP parameters
		cv.pkl       # CV splits
	
	<param_iteration>-<cv_iteration>
		config.json  # SP parameters
		stats.csv    # Time and accuracy using features
		p.pkl        # SP permanences
		syn_map.pkl  # SP synaptic map
		sp.pkl       # SP instance
		te_x.pkl     # SP output for each training sample
		tr_x.pkl     # SP output for each testing sample

####
# Local parsing
####

<base_directory>
	<param_iteration>-<cv_iteration>
		config.json  # SP parameters
		stats.csv    # Time and accuracy using features
		p.pkl        # SP permanences
		syn_map.pkl  # SP synaptic map
		sp.pkl       # SP instance
		te_x.pkl     # SP output for each training sample
		tr_x.pkl     # SP output for each testing sample
	
	cv_results.pkl   # Parameter names, parameter values, accuracies
	cv_clf.pkl       # Grid scores, best score, best params

####
# Storage
####

<param_iteration>, <cv_iteration>

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import json, os, pkgutil, csv, cPickle

# Third party imports
import numpy as np

# Program imports
from mHTM.parallel import create_runner, execute_runner

def save_results(results, p):
	"""
	Save the results.
	
	@param results: The array containing the data to save.
	
	@param p: The full path to where the data should be saved.
	"""
	
	with open(p, 'wb') as f:
		cPickle.dump(results, f, cPickle.HIGHEST_PROTOCOL)

def load_results(p):
	"""
	Load the results.
	
	@param p: The full path to where the data should be loaded from.
	"""
	
	with open(p, 'rb') as f:
		return np.array(cPickle.load(f))

def get_missing_results(bp):
	"""
	Find any missing results.
	
	@param bp: The full path to the directory containing the runs.
	
	@return: A list of the missing experiments.
	"""
	
	missing = []
	for p in sorted(os.listdir(bp)):
		# Only work with valid runs
		try:
			param_iteration, cv_iteration = [int(x) for x in p.split('-')]
		except ValueError:
			continue
		
		# Check to see if path exists
		p2 = os.path.abspath(os.path.join(bp, p))
		if not os.path.exists(os.path.join(p2, 'stats.csv')):
			missing.append(p2)
	
	return missing

def get_results(bp, score_method='column'):
	"""
	Get the results for the experiment.
	
	@param bp: The full path to the directory containing the runs.
	
	@param score_method: The method to used for computing the accuracy. This
	must be one the following: "column" - Uses the set of active columns to
	compute the accuracy, "prob" - Use the probabilistic version of the input,
	or "reduction" - Uses the dimensionality reduced version of the input.

	@return: The results and the base paths
	"""
	
	# Store the results
	results = []
	prev_param_iteration = None
	method = 'Accuracy: ' + score_method
	paths = []
	
	# Get the data
	for p in sorted(os.listdir(bp)):
		# Only work with valid runs
		try:
			param_iteration, cv_iteration = [int(x) for x in p.split('-')]
		except ValueError:
			continue
		
		# Read in the accuracy
		with open(os.path.join(bp, p, 'stats.csv')) as f:
			reader = csv.reader(f)
			for row in reader:
				key, value = row
				if method in key:
					accuracy = float(value)
		
		# Add to data structure
		if prev_param_iteration == param_iteration:
			results[-1].append(accuracy)
		else:
			prev_param_iteration = param_iteration
			paths.append(os.path.join(bp, p.split('-')[0]))
			results.append([accuracy])
	
	return np.array(results), paths

def get_top_paths(bp, k=10):
	"""
	Get the top paths for each method. This method uses the mean across the
	CV runs.
	
	@param bp: The full path to the directory containing the runs.
	
	@param k: Select this many maximum from the results.
	
	@return: A list of the paths with the top results.
	"""
	
	# Get top results for each type
	top_paths = set()
	methods = ('column', 'prob', 'reduction')
	for method in methods:
		accuracy, paths = get_results(bp, method)
		for ix in accuracy.mean(1).argsort()[::-1][:k]:
			top_paths.add(paths[ix])
	
	return sorted(top_paths)

def get_top_path(bp):
	"""
	Get the top path out of all methods. This method uses the mean across the
	CV runs.
	
	@param bp: The full path to the directory containing the runs.
	
	@param k: Select this many maximum from the results.
	
	@return: A string containing the top path. This is the best path out of the
	methods utilized.
	"""
	
	# Get top result
	best, best_path = 0., None
	methods = ('column', 'prob', 'reduction')
	for method in methods:
		accuracy, paths = get_results(bp, method)
		avg_accuracy = accuracy.mean(1)
		ix = avg_accuracy.argsort()[::-1][0]
		result, path = avg_accuracy[ix], paths[ix]
		if result > best:
			best, best_path = result, path
	
	return best_path

def launch_top_runs(top_paths, bp, command, auto_pupdate=False,
	partition_name='debug', time_limit='04-00:00:00', memory_limit=2048):
	"""
	Launch the top runs.
	
	@param top_paths: The full path to the base directory containing the top
	results.
	
	@param bp: The new base directory.
	
	@param command: The base command to execute in the runner. Two additional
	arguments will be passed - the base directory and the fold index.
	
	@param auto_pupdate: If True the permanence increment and decrement amounts
	will automatically be computed by the runner. If False, the ones specified
	in the config file will be used.
	
	@param partition_name: The partition name to use.
	
	@param time_limit: The maximum time limit.
	
	@param memory_limit: The maximum memory requirements in MB.
	"""
	
	for p in top_paths:
		# Path where the run should occur
		job_name = os.path.basename(p)
		p2 = os.path.join(bp, job_name)
		try:
			os.makedirs(p2)
		except OSError:
			pass # Overwrite the files
		
		# Create the runner
		runner_path = os.path.join(p2, 'runner.sh')
		command_new = '{0} "{1}" "{2}" {3}'.format(command, p, p2,
			int(auto_pupdate))
		stdio_path = os.path.join(p2, 'stdio.txt')
		stderr_path = os.path.join(p2, 'stderr.txt')
		create_runner(command=command_new, runner_path=runner_path,
			job_name=job_name, partition_name=partition_name,
			stdio_path=stdio_path, stderr_path=stderr_path,
			time_limit=time_limit, memory_limit=memory_limit)
		
		# Execute the runner
		execute_runner(runner_path)

def launch_missing(missing, command, partition_name='debug',
	time_limit='00-04:00:00', memory_limit=512):
	"""
	Launch the missing results on the cluster. It assumes that the convention
	<run_instance>-<fold_instance> for the directories was utilized.
	
	@param missing: The missing items.
	
	@param command: The base command to execute in the runner. Two additional
	arguments will be passed - the base directory and the fold index.
	
	@param partition_name: The partition name to use.
	
	@param time_limit: The maximum time limit.
	
	@param memory_limit: The maximum memory requirements in MB.
	"""
	
	# Execute each missing item
	for p in missing:
		# Build the SP kargs for the proper path
		bn, ix = os.path.basename(p).split('-')
		bp = os.path.join(os.path.dirname(p), bn)
		with open(os.path.join(bp, 'config.json'), 'rb') as f:
			kargs = json.load(f)
		kargs['log_dir'] = p
		
		# Dump the arguments to a new file
		s = json.dumps(kargs, sort_keys=True, indent=4,
			separators=(',', ': ')).replace('},', '},\n')
		with open(os.path.join(bp, 'config-{0}.json'.format(ix)), 'wb') as f:
			f.write(s)
		
		# Create the runner
		runner_path = os.path.join(bp, 'runner-{0}.sh'.format(ix))
		job_name = os.path.basename(p)
		command_new = '{0} "{1}" {2}'.format(command, bp, ix)
		stdio_path = os.path.join(bp, 'stdio-{0}.txt'.format(ix))
		stderr_path = os.path.join(bp, 'stderr-{0}.txt'.format(ix))
		create_runner(command=command_new, runner_path=runner_path,
			job_name=job_name, partition_name=partition_name,
			stdio_path=stdio_path, stderr_path=stderr_path,
			time_limit=time_limit, memory_limit=memory_limit)
		
		# Execute the runner
		execute_runner(runner_path)

if __name__ == '__main__':
	# Parameters to launch jobs
	mnist_runner_path = os.path.join(pkgutil.get_loader('mHTM.examples').
		filename, 'mnist_runner.py')
	command = 'python "{0}"'.format(mnist_runner_path)
	
	# Execute each type
	types = ('global', 'local')
	for type in types:	
		# Find and launch any missing jobs
		bp = 'results/full_mnist-{0}'.format(type)
		# missing = get_missing_results(bp)
		# if len(missing) > 0:
			# print missing
			# launch_missing(missing, command, 'work')
		
		# Launch the top jobs
		# bp2 = 'results/full_mnist_auto-{0}'.format(type)
		# top_paths = get_top_paths(bp)
		# launch_top_runs([top_path], bp2, command, 'work')
		
		# Find the top job
		print get_top_path(bp)