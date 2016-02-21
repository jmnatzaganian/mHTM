# boost_experiment.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 10/13/15
#	
# Description    : Study the boost.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Study the boost.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import cPickle, random, csv, os, time, json

# Third party imports
import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Program imports
from mHTM.region import SPRegion
from mHTM.plot import plot_error, compute_err

def make_data(p, nitems=100, width=100, density=0.9, seed=123456789):
	"""
	Make the dataset.
	
	@param p: the full path to where the dataset should be created.
	
	@param nitems: The number of items to create.
	
	@param width: The size of the input.
	
	@param density: The percentage of active bits.
	
	@param seed: The random number seed.
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

def load_data(p):
	"""
	Get the dataset.
	
	@param p: the full path to the dataset.
	"""
		
	with open(p, 'rb') as f:
		ds = cPickle.load(f)
	return ds

def _phase3(self):
	"""
	Normal phase 3, but with tracking the boost changes. Double commented lines
	are new.
	"""
	
	# Update permanences
	self.p = np.clip(self.p + (self.c_pupdate * self.y[:, 0:1] *
		self.x[self.syn_map] - self.pdec * self.y[:, 0:1]), 0, 1)
	
	if self.disable_boost is False:
		# Update the boosting mechanisms
		if self.global_inhibition:
			min_dc = np.zeros(self.ncolumns)
			min_dc.fill(self.c_mdc * bn.nanmax(self.active_dc))
		else:
			min_dc = self.c_mdc * bn.nanmax(self.neighbors * self.active_dc, 1)
		
		## Save pre-overlap boost info
		boost = list(self.boost)
		
		# Update boost
		self._update_active_duty_cycle()
		self._update_boost(min_dc)
		self._update_overlap_duty_cycle()
	
		## Write out overlap boost changes
		with open(os.path.join(self.out_path, 'overlap_boost.csv'), 'ab') as f:
			writer = csv.writer(f)
			writer.writerow([self.iter, bn.nanmean(boost != self.boost)])
	
		# Boost permanences
		mask = self.overlap_dc < min_dc
		mask.resize(self.ncolumns, 1)
		self.p = np.clip(self.p + self.c_sboost * mask, 0, 1)
	
		## Write out permanence boost info
		with open(os.path.join(self.out_path, 'permanence_boost.csv'), 'ab') \
			as f:
			writer = csv.writer(f)
			writer.writerow([self.iter, bn.nanmean(mask)])
	
	# Trim synapses
	if self.trim is not False:
		self.p[self.p < self.trim] = 0

def main(ds, p, ncols=2048, duty_cycle=100, nepochs=10, global_inhibition=True,
	seed=123456789):
	"""
	Run an experiment.
	
	@param ds: The dataset.
	
	@param p: The full path to the directory to save the results.
	
	@param ncols: The number of columns.
	
	@param duty_cycle: The duty cycle.
	
	@param nepochs: The number of epochs
	
	@param global_inhibition: If True use global inhibition otherwise use local
	inhibition.
	
	@param seed: The random seed.
	"""
	
	# Get some parameters
	ninputs = ds.shape[1]
	density = np.sum(ds[0]) / float(ninputs)
	
	# Make the directory if it doesn't exist
	try:
		os.makedirs(p)
	except OSError:
		pass
	
	# Initializations
	np.random.seed(seed)
	kargs = {
		'ninputs': ninputs,
		'ncolumns': ncols,
		'nsynapses': 40,
		'random_permanence': True,
		'pinc':0.03, 'pdec':0.05,
		'seg_th': 15,
		'nactive': int(0.02 * ncols),
		'duty_cycle': duty_cycle,
		'max_boost': 10,
		'global_inhibition': global_inhibition,
		'trim': 1e-4
	}
	
	# Create the region
	delattr(SPRegion, '_phase3')
	setattr(SPRegion, '_phase3', _phase3)
	sp = SPRegion(**kargs)
	sp.iter, sp.out_path = 1, p
	
	# Train the region
	t = time.time()
	for i in xrange(nepochs):
		for j, x in enumerate(ds):
			sp.execute(x)
			sp.iter += 1
	t = time.time() - t
	
	# Dump the details
	kargs['density'] = density
	kargs['seed'] = seed
	kargs['nepochs'] = nepochs
	kargs['time'] = t
	with open(os.path.join(p, 'details.json'), 'wb') as f:
		f.write(json.dumps(kargs, sort_keys=True, indent=4,
			separators=(',', ': ')))

def vary_density(bp, global_inhibition=True):
	"""
	Vary the density level.
	
	@pram bp: The base path.
	
	@param global_inhibition: If True use global inhibition otherwise use local
	inhibition.
	"""
	
	density_levels = np.linspace(.01, .99, 99)
	
	for density in density_levels:
		d = int(density * 100)
		print d
		p = os.path.join(bp, str(d))
		p2 = os.path.join(p, 'data.pkl')
		try:
			os.makedirs(p)
		except OSError:
			pass
		make_data(p2, density=density, seed=123456789)
		
		# Repeat for good results
		Parallel(n_jobs=-1)(delayed(main)(load_data(p2),
			os.path.join(p, str(i)), global_inhibition=global_inhibition,
			seed=i) for i in xrange(10))

def vary_dutycycle(bp, ds, global_inhibition=True):
	"""
	Vary the duty cycles.
	
	@pram bp: The base path.
	
	@param ds: The dataset to use.
	
	@param global_inhibition: If True use global inhibition otherwise use local
	inhibition.
	"""
	
	duty_cycles = (1, 10, 100, 1000, 10000)
	
	try:
		os.makedirs(bp)
	except OSError:
		pass

	for dc in duty_cycles:
		print '\n\n\n --------{0}-------- \n\n\n'.format(dc)
		p = os.path.join(bp, str(dc))		
		main(ds, p, duty_cycle=dc, nepochs=1,
			global_inhibition=global_inhibition)

def plot_density_results(bp, bp2=None):
	"""
	Average the results.
	
	@param bp: The base path.
	
	@param bp2: The second base path.
	"""
	
	def average(p):
		"""
		Compute the average activations for each density.
		
		@param p: The path to the file.
		
		@return: The average.
		"""
		
		with open(p, 'rb') as f:
			reader = csv.reader(f)
			data = []
			for row in reader:
				data.append(float(row[1]))
		return np.mean(data) * 100
	
	def get_data(p):
		"""
		Get the data for a single run.
		
		@param p: The path.
		
		@return: A tuple containing the overlap and permanences.
		"""
		
		overlap, permanence = [], []
		for d in os.listdir(p):
			npath = os.path.join(p, d)
			if os.path.isdir(npath):
				overlap.append(average(os.path.join(npath,
					'overlap_boost.csv')))
				permanence.append(average(os.path.join(npath,
					'permanence_boost.csv')))
		return np.array(overlap), np.array(permanence)
	
	def get_all_data(bp):
		"""
		Get the data for all runs.
		
		@param bp: The base path.
		
		@return: A tuple containing the sparsity, overlap, and permanences.
		"""
		
		overlap, permanence, sparsity = [], [], []
		for d in sorted([int(x) for x in os.listdir(bp)]):
			sparsity.append((1 - (d / 100.)) * 100)
			o, p = get_data(os.path.join(bp, str(d)))
			overlap.append(o)
			permanence.append(p)
		return np.array(sparsity[::-1]), np.array(overlap[::-1]), \
			np.array(permanence[::-1])
	
	def make_plot_params(sparsity, overlap, permanence, title=None):
		"""
		Generate the parameters for the plot.
		
		@param sparsity: The sparsity array.
		
		@param overlap: The overlap array.
		
		@param permanence: The permanence array.
		
		@param title: The title for the plot.
		
		@return: A dictionary with the parameters.
		"""
		
		return {'x_series':(sparsity, sparsity),
			'y_series':(np.median(overlap, 1), np.median(permanence, 1)),
			'series_names':('Overlap Boosting', 'Permanence Boosting'),
			'y_errs':(compute_err(overlap), compute_err(permanence)),
			'xlim':(0, 100), 'ylim':(0, 30), 'title':title
			}
	
	data = get_all_data(bp)
	if bp2 is None:
		plot_error(**make_plot_params(*data))
	else:
		# Make main plot
		fig = plt.figure(figsize=(21, 20), facecolor='white')
		ax = fig.add_subplot(111)
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.tick_params(labelcolor='w', top='off', bottom='off', left='off',
			right='off')
		ax.set_xlabel('Sparsity [%]')
		ax.set_ylabel('% Columns Boosted')
		
		# Make subplots
		ax1 = fig.add_subplot(211)
		plot_error(show=False, legend=False, ax=ax1, **make_plot_params(*data,
			title='Global Inhibition'))
		data2 = get_all_data(bp2)
		ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
		plot_error(show=False, ax=ax2, **make_plot_params(*data2,
			title='Local Inhibition'))
		
		# Save it
		plt.subplots_adjust(bottom=0.15, hspace=0.3)
		plt.savefig('boost_sparseness.png', format='png',
			facecolor=fig.get_facecolor(), edgecolor='none')

def plot_single_run(bp):
	"""
	Create an error plot for a single run.
	
	@param bp: The base path.
	"""
	
	def read(p):
		"""
		Read in the data.
		
		@param p: The path to the file to read.
		
		@return: The results.
		"""
		
		with open(p, 'rb') as f:
			reader = csv.reader(f)
			data = []
			for row in reader:
				data.append(float(row[1]))
		return np.array(data) * 100
	
	def get_data(p):
		"""
		Get all of the results.
		
		@param p: The directory to obtain the data in.
		
		@return: The results.
		"""
		
		permanence = []
		for d in os.listdir(p):
			npath = os.path.join(p, d)
			if os.path.isdir(npath):
				permanence.append(read(os.path.join(npath,
					'permanence_boost.csv')))
		return np.array(permanence)
	
	data = get_data(bp)
	plot_error(show=True,
		x_series=(np.arange(data.shape[1]),),
		y_series=(np.median(data, 0),),
		y_errs=(compute_err(data, axis=0), ),
		xlim=(0, 20), ylim=(0, 100),
		x_label='Iteration', y_label='% Columns Boosted'
		)

if __name__ == '__main__':
	# Params
	base_dir = os.path.join(os.path.expanduser('~'), 'scratch')
	p1 = os.path.join(base_dir, 'boost_experiments-global')
	p2 = os.path.join(base_dir, 'boost_experiments-local')
	
	# Experiment
	# vary_density(p1, True)
	# vary_density(p2, False)
	plot_density_results(p1, p2)
