# parser.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 03/20/16
# 
# Description    : Module for parsing the mnist novelty detection results.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
 odule for parsing the mnist novelty detection results.

####
# Cluster parsing
####

<base_directory>
	<param_iteration>
		config.json  # SP parameters
	
	<param_iteration>-<iteration>
		config.json  # SP parameters
		stats.csv    # Time and accuracy using features
		p.pkl        # SP permanences
		syn_map.pkl  # SP synaptic map
		sp.pkl       # SP instance
		te_x.pkl     # SP output for each training sample
		tr_x.pkl     # SP output for each testing sample

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import os, csv

# Third party imports
import numpy as np

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

def get_results(bp):
	"""
	Get the results for the experiment.
	
	@param bp: The full path to the directory containing the runs.

	@return: The results and the base paths
	"""
	
	# Store the results
	results = []
	prev_param_iteration = None
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
				if 'SP % Adjusted Score' == key:
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
	Get the top paths out of all of the runs.
	
	@param bp: The full path to the directory containing the runs.
	
	@param k: Select this many maximum from the results.
	
	@return: A list of the paths with the top results.
	"""
	
	# Get top results for each type
	top_paths = set()
	accuracy, paths = get_results(bp, method)
	for ix in accuracy.mean(1).argsort()[::-1][:k]:
		top_paths.add(paths[ix])
	
	return sorted(top_paths)

def get_top_path(bp):
	"""
	Get the top path out of all of the runs.
	
	@param bp: The full path to the directory containing the runs.
	
	@param k: Select this many maximum from the results.
	
	@return: A tuple containing the best path and its corresponding accuracy.
	"""
	
	# Get top result
	best, best_path = 0., None
	accuracy, paths = get_results(bp)
	avg_accuracy = accuracy.mean(1)
	ix = avg_accuracy.argsort()[::-1][0]
	result, path = avg_accuracy[ix], paths[ix]
	if result > best:
		best, best_path = result, path
	
	return best_path, accuracy

if __name__ == '__main__':
	# Find any missing jobs
	bp = 'results/mnist_novelty_detection'
	missing = get_missing_results(bp)
	if len(missing) > 0:
		for item in missing:
			print item
		
	# Find the top job
	print get_top_path(bp)
