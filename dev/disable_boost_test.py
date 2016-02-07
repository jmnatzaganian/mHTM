# disable_boost_test
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 02/06/16
#	
# Description    : Test for disabling the boost.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Test for disabling the boost.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import cPickle, random, os, time, json

# Third party imports
import numpy as np

# Program imports
from mHTM.region import SPRegion

def make_data(p, nitems=100, width=100, density=0.3, seed=123456789):
	"""
	Make the dataset.
	
	@param p: the full path to where the dataset should be created.
	
	@param nitems: The number of items to create.
	
	@param width: The size of the input.
	
	@param density: The percentage of active bits.
	
	@param seed: The random number seed.
	
	@return: The dataset.
	"""
	
	# Initialization
	random.seed(seed)
	np.random.seed(seed)
	nactive = int(width * density)
	
	# Build the dataset
	ds = np.zeros((nitems, width), dtype='bool')
	for i in xrange(nitems):
		indexes = set(np.random.randint(0, width, nactive))
		while len(indexes) != nactive:
			indexes.add(random.randint(0, width - 1))
		ds[i][list(indexes)] = True
	
	# Write the file
	with open(p, 'wb') as f:
		cPickle.dump(ds, f, cPickle.HIGHEST_PROTOCOL)
	
	return ds

def load_data(p):
	"""
	Get the dataset.
	
	@param p: the full path to the dataset.
	"""
		
	with open(p, 'rb') as f:
		ds = cPickle.load(f)
	return ds

def main(ds, p, n_iters=100, seed=123456789):
	"""
	Run an experiment.
	
	@param ds: The dataset.
	
	@param p: The full path to the directory to save the results.
	
	@param n_iters: Number of iterations to run the experiment.
	
	@param seed: The random seed.
	"""
	
	def go(kargs, type):
		"""
		Execute SP.
		
		@param kargs: The params for the SP.
		
		@param type: Run type.
		
		@return: The new sp and the execution time.
		"""
		
		# Run the SP
		t = time.time()
		for _ in xrange(n_iters):
			sp = SPRegion(**kargs)
			sp.c_sboost = 0 # Ensure that no permanence boosting occurs
			sp.execute(ds, store=False)
		t = time.time() - t
		
		# Dump the permanence matrix
		with open(os.path.join(p, '{0}-permanence.pkl'.format(type)), 'wb') \
			as f:
			cPickle.dump(sp.p, f, cPickle.HIGHEST_PROTOCOL)
		
		# Dump the details
		kargs['density'] = density
		kargs['seed'] = seed
		kargs['time'] = t
		with open(os.path.join(p, '{0}-details.json'.format(type)), 'wb') as f:
			f.write(json.dumps(kargs, sort_keys=True, indent=4,
				separators=(',', ': ')))
		
		return sp, t
	
	# Get some parameters
	ninputs = ds.shape[1]
	density = np.sum(ds[0]) / float(ninputs)
	
	# Make the directory if it doesn't exist
	try:
		os.makedirs(p)
	except OSError:
		pass
	
	# Build the params
	b_kargs = {
		'ninputs': ninputs,
		'ncolumns': 300,
		'nsynapses': 40,
		'random_permanence': True,
		'pinc':0.03, 'pdec':0.05,
		'seg_th': 15,
		'nactive': int(0.02 * 300),
		'max_boost': 1,
		'global_inhibition': True,
		'trim': 1e-4,
		'seed': seed,
	}
	nb_kargs = b_kargs.copy()
	nb_kargs['disable_boost'] = True
	
	# Train each SP - give boosting SP cache advantage
	nb_sp, nb_t = go(nb_kargs, 'noboost')
	b_sp, b_t = go(b_kargs, 'boost')
	
	# Verify that the two are the same
	print 'Results are identical:', np.all(b_sp.p == nb_sp.p)
	print 'Speedup if boosting is disabled:', b_t / nb_t

if __name__ == '__main__':
	# Path to some scratch space to work with - default to home directory
	p = os.path.join(os.path.expanduser('~'), 'scratch')
	
	# Build the data
	try:
		ds = load_data(os.path.join(p, 'ds.pkl'))
	except IOError:
		ds = make_data(os.path.join(p, 'ds.pkl'))
	
	# Run the experiment
	main(ds, p)