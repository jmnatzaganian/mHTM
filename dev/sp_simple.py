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

# Program imports
from mHTM.region import SPRegion
from mHTM.datasets.loader import SPDataset
from mHTM.metrics import SPMetrics

def main():
	"""
	Program entry.
	
	Build an SP using SPDataset and see how it performs.
	"""
	
	# Params
	nsamples, nbits, pct_active = 500, 100, 0.4
	ncolumns = 300
	base_path = os.path.join(os.path.expanduser('~'), 'scratch', 'sp_simple')
	seed = 532833024
	kargs = {
		'ninputs': nbits,
		'ncolumns': 300,
		'nactive': 0.02 * ncolumns,
		'global_inhibition': True,
		'trim': 1e-4,
		'disable_boost': True,
		'seed': seed,
		
		'nsynapses': 20,
		'seg_th': 2,
		
		'syn_th': 0.5,
		'pinc': 0.01,
		'pdec': 0.01,
		'pwindow': 0.5,
		'random_permanence': True,
		
		'nepochs': 1,
		'log_dir': os.path.join(base_path, '1-1')
	}
	
	metrics = SPMetrics()
	pct_noise = .15
	ds = SPDataset(nsamples=nsamples, nbits=nbits, pct_active=pct_active,
			pct_noise=pct_noise, seed=seed)
	sp = SPRegion(**kargs)
	sp.fit(ds.data)
	sp_output = sp.predict(ds.data)
	print metrics.compute_uniqueness(sp_output)	
	
	with open(os.path.join(base_path, 'ds_data.pkl'), 'wb') as f:
		cPickle.dump(ds.data, f, cPickle.HIGHEST_PROTOCOL)
	with open(os.path.join(base_path, 'perm.pkl'), 'wb') as f:
		cPickle.dump(sp.p, f, cPickle.HIGHEST_PROTOCOL)
	with open(os.path.join(base_path, 'sp_output.pkl'), 'wb') as f:
		cPickle.dump(sp_output, f, cPickle.HIGHEST_PROTOCOL)

def main2():
	base_dir = r'C:\Users\james\scratch\sp_simple'
	file = 'sp_output'
	
	with open(os.path.join(base_dir, '{0}.pkl'.format(file)), 'rb') as f:
		local = cPickle.load(f)
	with open(os.path.join(base_dir, '{0}_2.pkl'.format(file)), 'rb') as f:
		remote = cPickle.load(f)
	
	print np.all(local == remote)
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	main()
	# main2()
