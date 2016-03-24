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
import os

# Third party imports
import numpy as np

# Program imports
from mHTM.region import SPRegion
from mHTM.datasets.loader import SPDataset
from mHTM.metrics import SPMetrics
from mHTM.plot import plot_line

def main():
	"""
	Program entry.
	
	Build an SP using SPDataset and see how it performs.
	"""
	
	# Params
	nsamples, nbits, pct_active = 500, 100, 0.4
	base_path = os.path.join(os.path.expanduser('~'), 'scratch', 'sp_simple')
	seed = 532833024
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
		'log_dir': os.path.join(base_path, '1-1')
	}
	
	# Build items to store results
	npoints = 25
	pct_noises = np.linspace(0, 1, npoints, False)
	uniqueness_sp, uniqueness_data = np.zeros(npoints), np.zeros(npoints)
	overlap_sp, overlap_data = np.zeros(npoints), np.zeros(npoints)
	
	# Metrics
	metrics = SPMetrics()
		
	# Vary input noise
	for i, pct_noise in enumerate(pct_noises):
		print 'Iteration {0} of {1}'.format(i + 1, npoints)
		
		# Build the dataset
		ds = SPDataset(nsamples=nsamples, nbits=nbits, pct_active=pct_active,
			pct_noise=pct_noise, seed=seed)
		
		# Get the dataset stats
		uniqueness_data[i] = metrics.compute_uniqueness(ds.data)
		overlap_data[i] = metrics.compute_overlap(ds.data)
		
		# Build the SP
		sp = SPRegion(**kargs)
		
		# Train the region
		sp.fit(ds.data)
		
		# Get the SP's output SDRs
		sp_output = sp.predict(ds.data)
		
		# Get the stats
		uniqueness_sp[i] = metrics.compute_uniqueness(sp_output)
		overlap_sp[i] = metrics.compute_overlap(sp_output)
	
	# Make some plots
	print 'Showing uniqueness - 0% is ideal'
	plot_line([pct_noises * 100, pct_noises * 100], [uniqueness_data * 100,
		uniqueness_sp * 100], series_names=('Raw Data', 'SP Output'),
		x_label='% Noise', y_label='Uniqueness [%]', xlim=False,
		ylim=(-5, 105),	out_path=os.path.join(base_path, 'uniqueness.png'),
		show=True)
	print 'Showing average normalized overlap - 100% is ideal'
	plot_line([pct_noises * 100, pct_noises * 100], [overlap_data * 100,
		overlap_sp * 100], series_names=('Raw Data', 'SP Output'),
		x_label='% Noise', y_label="% Normalized Overlap", xlim=False,
		ylim=(-5, 105), out_path=os.path.join(base_path, 'overlap.png'),
		show=True)
	
	print '*** All data saved in "{0}" ***'.format(base_path)

if __name__ == '__main__':
	main()
