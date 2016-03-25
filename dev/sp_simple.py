# sp_simple.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 12/13/15
#	
# Description    : Simple demonstration of SP showing metric usage.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Simple demonstration of SP showing metric usage.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import os, cPickle

# Third party imports
import numpy as np
from joblib import Parallel, delayed

# Program imports
from mHTM.region import SPRegion
from mHTM.datasets.loader import SPDataset
from mHTM.metrics import SPMetrics
from mHTM.plot import plot_error, compute_err

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

def base_experiment(log_dir, seed = 123456789):
	"""
	The base experiment.
	
	Build an SP using SPDataset and see how it performs.
	
	@param log_dir: The full path to the log directory.
	
	@param seed: The random seed to use.
	
	@return: Tuple containing: SP uniqueness, input uniqueness, SP overlap,
	input overlap.
	"""
	
	# Params
	nsamples, nbits, pct_active = 500, 100, 0.4
	kargs = {
		'ninputs': nbits,
		'ncolumns': 200,
		'nactive': 50,
		'global_inhibition': True,
		'trim': 1e-4,
		'disable_boost': True,
		'seed': seed,
		
		'nsynapses': 75,
		'seg_th': 15,
		
		'syn_th': 0.5,
		'pinc': 0.001,
		'pdec': 0.001,
		'pwindow': 0.5,
		'random_permanence': True,
		
		'nepochs': 10,
		'log_dir': log_dir
	}
	
	# Seed numpy
	np.random.seed(seed)
	
	# Build items to store results
	npoints = 101
	pct_noises = np.linspace(0, 1, npoints)
	u_sp, u_ip = np.zeros(npoints), np.zeros(npoints)
	o_sp, o_ip = np.zeros(npoints), np.zeros(npoints)
	
	# Metrics
	metrics = SPMetrics()
		
	# Vary input noise
	for i, pct_noise in enumerate(pct_noises):
		# Build the dataset
		ds = SPDataset(nsamples=nsamples, nbits=nbits, pct_active=pct_active,
			pct_noise=pct_noise, seed=seed)
		x = ds.data
		
		# Get the dataset stats
		u_ip[i] = metrics.compute_uniqueness(x) * 100
		o_ip[i] = metrics.compute_overlap(x) * 100
		
		# Build the SP
		sp = SPRegion(**kargs)
		
		# Train the region
		sp.fit(x)
		
		# Get the SP's output SDRs
		sp_output = sp.predict(x)
		
		# Get the stats
		u_sp[i] = metrics.compute_uniqueness(sp_output) * 100
		o_sp[i] = metrics.compute_overlap(sp_output) * 100
		
		# Log everything
		sp._log_stats('% Input Uniqueness', u_ip[i])
		sp._log_stats('% Input Overlap', o_ip[i])
		sp._log_stats('% SP Uniqueness', u_sp[i])
		sp._log_stats('% SP Overlap', o_sp[i])
	
	return u_sp, u_ip, o_sp, o_ip

def main(base_path, ntrials=10, seed=123456789):
	"""
	Run the experiments.
	
	@param base_path: The full path to where the data should be stored.
	
	@param ntrials: The number of trials to use for the experiment.
	
	@param seed: The seed for the random number generator.
	"""
	
	# X-Axis data
	npoints = 101
	pct_noises = np.linspace(0, 1, npoints)
	x = (pct_noises * 100, pct_noises * 100)
	
	# Run the experiment
	results = Parallel(n_jobs=-1)(delayed(base_experiment)(
		os.path.join(base_path, 'run-{0}'.format(i)), seed2) for i, seed2 in
		enumerate(generate_seeds(ntrials, seed), 1))
	u_sp = np.zeros((len(results), npoints))
	u_ip = np.zeros((len(results), npoints))
	o_sp = np.zeros((len(results), npoints))
	o_ip = np.zeros((len(results), npoints))
	for i, (a, b, c, d) in enumerate(results):
		u_sp[i], u_ip[i], o_sp[i], o_ip[i] = a, b, c, d
	
	# Save the results
	with open(os.path.join(base_path, 'results.pkl'), 'wb') as f:
		cPickle.dump((u_sp, u_ip, o_sp, o_ip), f, cPickle.HIGHEST_PROTOCOL)
	
	# Make some plots
	e = (compute_err(u_sp, axis=0), compute_err(u_ip, axis=0))
	y = (np.median(u_sp, 0), np.median(u_ip, 0))
	plot_error(x, y, ('SP Output', 'Raw Data'), e, '% Noise', 'Uniqueness [%]',
		xlim=False,	ylim=(-5, 105),	out_path=os.path.join(base_path,
		'uniqueness.png'), show=False)
	e = (compute_err(o_sp, axis=0), compute_err(o_ip, axis=0))
	y = (np.median(o_sp, 0), np.median(o_ip, 0))
	plot_error(x, y, ('SP Output', 'Raw Data'), e, '% Noise',
		'Normalized Overlap [%]', xlim=False, ylim=(-5, 105),
		out_path=os.path.join(base_path, 'overlap.png'), show=False)

if __name__ == '__main__':
	base_path = os.path.join(os.path.expanduser('~'), 'scratch', 'sp_simple')
	main(base_path)
