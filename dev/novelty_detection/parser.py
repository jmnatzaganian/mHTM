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

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import os, cPickle, re

# Third party imports
import numpy as np
import pandas as pd

# Program imports
from mHTM.plot import plot_surface, plot_error, compute_err

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

def make_3d_plot(p):
	"""
	Make a nice looking 3D plot.
	
	@param p: The full path to the results file.
	"""
	
	# Get the data
	with open(p, 'rb') as f:
		sp_x, sp_y, svm_x, svm_y, param = cPickle.load(f)
	sp_x, sp_y = np.median(sp_x, 1), np.median(sp_y, 1)
	svm_x, svm_y = np.median(svm_x, 1), np.median(svm_y, 1)
	
	# Fix the messed up sort order
	ix = np.array(pd.DataFrame(param).sort_values([0, 1],
		ascending=[True, True]).index).astype('i')
	param = param[ix]
	sp_x = sp_x[ix]
	sp_y = sp_y[ix]
	svm_x = svm_x[ix]
	svm_y = svm_y[ix]
	
	# Refactor the data
	x, y, z = [], [], []
	i = 0
	while i < len(param):
		xi, yi, zi = [], [], []
		xii, yii = param[i]
		p_xii = xii
		while p_xii == xii:
			xi.append(xii)
			yi.append(yii)
			zi.append(i)
			i += 1
			if i >= len(param):
				max_overlap = yii
				break
			xii, yii = param[i]
		x.append(xi)
		y.append(yi)
		z.append(zi)
	x = np.array(x).T * 100
	y = (np.array(y).T / max_overlap) * 100	
	z = np.array(z).T
	
	# Make the plots
	dir = os.path.dirname(p)
	plot_surface(x, y, sp_x[z], '% Noise', '% Overlap', '\n% Error',
		show=False, out_path=os.path.join(dir, 'sp_tr.png'))
	plot_surface(x, y, sp_y[z], '% Noise', '% Overlap', '\n% Error',
		show=False, out_path=os.path.join(dir, 'sp_te.png'))
	plot_surface(x, y, svm_x[z], '% Noise', '% Overlap', '\n% Error',
		show=False, out_path=os.path.join(dir, 'svm_tr.png'))
	plot_surface(x, y, svm_y[z], '% Noise', '% Overlap', '\n% Error',
		show=False, out_path=os.path.join(dir, 'svm_te.png'))

def make_2d_plot(p, noise=None, overlap=None):
	"""
	Make a 2D plot for the specified slice.
	
	@param p: The full path to the results file.
	
	@param noise: The degree of noise to use, if none all are used.
	
	@param overlap: The degree of overlap to use, if none all are used.
	"""
	
	# Get the data
	with open(p, 'rb') as f:
		sp_x, sp_y, svm_x, svm_y, param = cPickle.load(f)
	
	# Fix the messed up sort order
	ix = np.array(pd.DataFrame(param).sort_values([0, 1],
		ascending=[True, True]).index).astype('i')
	param = param[ix]
	sp_x, sp_y = sp_x[ix], sp_y[ix]
	svm_x, svm_y = svm_x[ix], svm_y[ix]
	
	# Refactor the data
	ix = []
	if noise is not None:
		id = 0
		id2 = 1
		val = noise
		func = float
		name = 'overlap'
		term = 100. / 40
	elif overlap is not None:
		id = 1
		id2 = 0
		val = overlap
		func = int
		name = 'noise'
		term = 100.
	else:
		raise 'noise and overlap are exclusive parameters'
	for i in xrange(len(param)):
		if func(param[i][id]) == val: ix.append(i)
	x = param[ix][:, id2] * term
	
	# Make the plots
	dir = os.path.dirname(p)
	plot_error((x, x), (np.median(sp_x[ix], 1), np.median(svm_x[ix], 1)),
		('SP', 'SVM'), (compute_err(sp_x[ix]), compute_err(svm_x[ix])),
		'% {0}'.format(name.capitalize()), '% Error', xlim=(-5, 105),
		ylim=(-5, 105), show=False,
		out_path=os.path.join(dir, 'train-{0}.png'.format(name)))
	plot_error((x, x), (np.median(sp_y[ix], 1), np.median(svm_y[ix], 1)),
		('SP', 'SVM'), (compute_err(sp_y[ix]), compute_err(svm_y[ix])),
		'% {0}'.format(name.capitalize()), '% Error', xlim=(-5, 105),
		ylim=(-5, 105), show=False,
		out_path=os.path.join(dir, 'test-{0}.png'.format(name)))

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
	
	# Make the pretty 3D plot
	p2 = os.path.join(user_path, 'full_results.pkl')
	make_3d_plot(p2)
	make_2d_plot(p2, overlap=0)
	make_2d_plot(p2, noise=0.35)
