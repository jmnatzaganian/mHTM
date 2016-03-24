# parser.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 03/23/16
# 
# Description    : Module for parsing the novelty detection results.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Module for parsing the novelty detection results.

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
import os, cPickle, re

# Third party imports
import numpy as np

def natural_sort(items):
	"""
	Sort a set of strings in the format that a human would.
	
	@param items: The list of items to sort.
	
	@return: A new list with the sorted items.
	"""
	
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key : [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(items, key = alphanum_key)

def get_missing(bp):
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

def dump_results(bp):
	"""
	Dump the results for the experiment.
	
	@param bp: The full path to the directory containing the runs.

	@return: The results and the base paths
	"""
	
	# Store the results
	sp_x_results = []
	sp_y_results = []
	svm_x_results = []
	svm_y_results = []
	param = []
	
	# Get the data
	for p in natural_sort(os.listdir(bp)):
		# Only work with valid runs
		try:
			noise, overlap = p.split('-')
		except ValueError:
			continue
		param.append([float(noise), int(overlap)])
		
		# Read in the data
		with open(os.path.join(bp, p, 'results.pkl')) as f:
			sp_x, sp_y, svm_x, svm_y = cPickle.load(f)
		
		# Add to data structures
		sp_x_results.append(sp_x)
		sp_y_results.append(sp_y)
		svm_x_results.append(svm_x)
		svm_y_results.append(svm_y)
	
	# Dump the results
	with open(os.path.join(bp, 'full_results.pkl'), 'wb') as f:
		cPickle.dump((np.array(sp_x_results), np.array(sp_y_results),
			np.array(svm_x_results), np.array(svm_y_results), np.array(param)),
			f, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	# Find any missing jobs
	user_path = os.path.expanduser('~')
	p = os.path.join(user_path, 'results', 'novelty_detection')
	
	missing = get_missing(p)
	if len(missing) > 0:
		for item in missing:
			print item
	else:
		# Repackage everything
		dump_results(p)
