# parser.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 02/27/16
# 
# Description    : Module for parsing the parameter exploration results.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Module for parsing the parameter exploration results.

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

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import csv, json, os, re, cPickle

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Program imports
from mHTM.plot import plot_error, compute_err

def natural_sort(items):
	"""
	Sort a set of strings in the format that a human would.
	
	@param items: The list of items to sort.
	
	@return: A new list with the sorted items.
	"""
	
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key : [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(items, key = alphanum_key)

def get_sorted_dirs(paths):
	"""
	Get all of the directories, sorted from the provided set of paths.
	
	@param paths: The paths to sort.
	
	@return: The sorted directories.
	"""
	
	return natural_sort([path for path in paths if os.path.isdir(path)])

def get_results(base_path, config_keys, x=None, y=None):
	"""
	Get the results for the experiment.
	
	@param base_path: The full path to the directory containing the runs.
	
	@param config_keys: The keys in the config to read.
	
	@param x: If not None add the data to this structure.
	
	@param y: If not None add the data to this structure.

	@return: The results and the base paths
	"""
	
	# Store the independent variables
	if x is None: x = [[] for _ in config_keys]
	
	# Store the dependent variables
	# -- fit_time, learn_fit_time, pred_fit_time, input_uniqueness,
	#	 input_overlap, input_correlation, sp_uniqueness, sp_overlap,
	#	 sp_correlation
	if y is None: y = [[],[],[],[],[],[],[],[],[]]
	
	# Get the data
	prev_param_iteration = None
	for path in sorted(os.listdir(base_path)):
		# Only work with valid runs
		try:
			param_iteration, _ = [int(item) for item in path.split('-')]
		except ValueError:
			continue
		
		#####
		# Independent variables
		#####
		
		# Get the JSON config
		with open(os.path.join(base_path, path, 'config.json'), 'rb') as f:
			config = json.load(f)
		
		# Get the data
		for i, key in enumerate(config_keys):
			x[i].append(config[key])
		
		#####
		# Dependent variables
		#####
		
		# Read in the results
		data = []
		with open(os.path.join(base_path, path, 'stats.csv'), 'rb') as f:
			reader = csv.reader(f)
			for row in reader: data.append(float(row[1]))
		
		# Add to data structure
		if prev_param_iteration == param_iteration:
			for i, d in enumerate(data): y[i][-1].append(d)
		else:
			prev_param_iteration = param_iteration
			for i, d in enumerate(data): y[i].append([d])
	
	return x, y

def main(root_dir):
	"""
	Parse out the experiment data into a user-friendly format.
	
	@param root_dir: The root of the directory tree.
	
	CAUTION: Known bug - If only one folder exists, this code will not produce
	the output.
	"""
	
	# Experiment map
	# -- Folder name, parameter names, experiment name
	experiment_map = {
		'nactive': [['nactive'], 'nactive'],
		'ncols1': [['ncolumns'], 'ncols'],
		'ncols2': [['ncolumns'], 'ncols'],
		'ncols3': [['ncolumns'], 'ncols'],
		'ncols4': [['ncolumns'], 'ncols'],
		'ncols5': [['ncolumns'], 'ncols'],
		'ncols6': [['ncolumns'], 'ncols'],
		'ncols7': [['ncolumns'], 'ncols'],
		'ncols8': [['ncolumns'], 'ncols'],
		'ncols9': [['ncolumns'], 'ncols'],
		'ncols10': [['ncolumns'], 'ncols'],
		'nepochs': [['nepochs'], 'nepochs'],
		'nsynapses': [['nsynapses'], 'nsynapses'],
		'pct_active': [['pct_active'], 'pct_active'],
		'pdec': [['pdec'], 'pdec'],
		'pinc': [['pinc'], 'pinc'],
		'pwindow': [['pwindow'], 'pwindow'],
		'seg_th': [['seg_th'], 'seg_th']
	}
	
	# Initial name of base experiment
	prev_name = None
	x = y = None
	
	# Process all of the items
	for dir in get_sorted_dirs([os.path.join(root_dir, p) for p in
		os.listdir(root_dir)]):
		parameter_names, experiment_name = experiment_map[
			os.path.basename(dir)]
		
		# Get the data
		if experiment_name == prev_name:
			x, y = get_results(dir, parameter_names, x, y)
		else:
			# Save the results
			if not ((x is None) and (y is None)):
				with open(os.path.join(root_dir, '{0}.pkl'.format(prev_name)), 'wb') as f:
					cPickle.dump((x, y), f, cPickle.HIGHEST_PROTOCOL)
			
			# Get the new data
			x, y = get_results(dir, parameter_names)
			prev_name = experiment_name

def main2(base_path):
	"""
	@param base_path: Full path to pickle file to work with.
	"""
	
	# Mapping of independent variables to indexes
	data_index = {
		'fit_time':0,
		'learn_fit_time':1,
		'pred_fit_time':2,
		'input_uniqueness':3,
		'input_overlap':4,
		'input_correlation':5,
		'sp_uniqueness':6,
		'sp_overlap':7,
		'sp_correlation':8
	}
	
	# Get the data
	with open(base_path, 'rb') as f:
		x, y = cPickle.load(f)
	x = sorted(set(x[-1])) # For now work with 1D
	
	# Pull out data for this plot
	y1 = (y[data_index['input_uniqueness']], y[data_index['sp_uniqueness']])
	y2 = (y[data_index['input_overlap']], y[data_index['sp_overlap']])
	y3 = (y[data_index['input_correlation']], y[data_index['sp_correlation']])
	
	# Refactor the data
	x_series = (x, x, x)
	med = lambda y: np.median(y, axis=1) * 100
	err = lambda y: compute_err(y, axis=1) * 100
	y1_series = map(med, y1)
	y1_errs = map(err, y1)
	y2_series = map(med, y2)
	y2_errs = map(err, y2)
	y3_series = map(med, y3)
	y3_errs = map(err, y3)
	
	# Make the main plot
	fig = plt.figure(figsize=(21, 20), facecolor='white')
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off',
		right='off')
	
	# Make subplots
	ax1 = fig.add_subplot(311)
	plot_error(show=False, legend=False, ax=ax1, title='Uniqueness',
		x_series=x_series, y_series=y1_series, y_errs=y1_errs, ylim=(-5, 105))
	ax2 = fig.add_subplot(312, sharex=ax1, sharey=ax1)
	plot_error(show=False, legend=False, ax=ax2, title='Overlap',
		x_series=x_series, y_series=y2_series, y_errs=y2_errs, ylim=(-5, 105))
	ax3 = fig.add_subplot(313, sharex=ax1, sharey=ax1)
	plot_error(show=False, legend=False, ax=ax3, title='Correlation',
		x_series=x_series, y_series=y3_series, y_errs=y3_errs, ylim=(-5, 105))
	plt.tight_layout(h_pad=2)
	plt.show()

if __name__ == '__main__':
	####
	# Parse
	####
	
	# results_dir = os.path.join(os.path.expanduser('~'), 'results')
	# experiment_name = 'first_order'
	# inhibition_types = ('global', 'local')
	# for inhibition_type in inhibition_types:
		# root_dir = os.path.join(results_dir, experiment_name, inhibition_type)
		# main(root_dir)

	####
	# Plot
	####
	
	results_dir = os.path.join(os.path.expanduser('~'), 'scratch')
	experiment_name = 'first_order'
	inhibition_type = 'global'
	root_dir = os.path.join(results_dir, experiment_name, inhibition_type)
	for experiment in (os.listdir(root_dir)):
		print experiment
		base_path = os.path.join(root_dir, experiment)
		main2(base_path)
