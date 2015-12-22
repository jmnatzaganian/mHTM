# region.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 10/12/15
#	
# Description    : Module for implementing an HTM region.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Implementation for a region in an HTM.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import os, time, cPickle, csv, json
from itertools import izip

# Third party imports
import numpy as np
import bottleneck as bn

def load_htm(path):
	"""
	Load an HTM.
	
	@param path: The full path to where the saved HTM is located.
	
	@return: An HTM object corresponding to the previously saved one.
	"""
	
	with open(path, 'rb') as f:
		htm = cPickle.load(f)
	return htm

def load_pkl(path):
	"""
	Load previously saved pickled data from the specified path.
	
	@param path: The full path to where the data is located.
	
	@return: The saved data.
	"""
	
	with open(path, 'rb') as f:
		data = cPickle.load(f)
	return data

class Region(object):
	"""
	Base class for an HTM region.
	"""
	
	@property
	def learn(self):
		"""
		Get the learning state.
		"""
		
		return self._learn
	
	@learn.setter
	def learn(self, learn):
		"""
		Set the learning state.
		"""
		
		self._learn = learn
		
	def save(self, name):
		"""
		Save the HTM. The save path will be in the log directory with the
		given name in a pickle file. Note that the ".pkl" extension will
		automatically be applied.
		
		@param name: The name of this HTM instance.
		"""
		
		with open(os.path.join(self.log_dir, name + '.pkl'), 'wb') as f:
			cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
	
	def _save_data(self, name, data):
		"""
		Save some data. The save path will be in the log directory with the
		given name in a pickle file. Note that the ".pkl" extension will
		automatically be applied.
		
		@param name: The name of the data being saved.
		
		@param data: The data to save.
		"""
		
		with open(os.path.join(self.log_dir, name + '.pkl'), 'wb') as f:
			cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)
	
	def _log_stats(self, name, value):
		"""
		Write the log data to a CSV file.
		
		@param name: The name of the item being timed.
		
		@param value: The value to log.
		"""
		
		with open(os.path.join(self.log_dir, 'stats.csv'), 'ab') as f:
			writer = csv.writer(f)
			writer.writerow([name, value])

class SPRegion(Region):
	"""
	Class for implementing a Spatial Pooler region. This implementation follows
	Numenta's original CLA whitepaper definition. It additionally meets the
	scikit-learn requirements. It also supports serialization.
	"""
	
	def __init__(self,
		# Parameters for the region
		ninputs, ncolumns=1024, pct_active=None, nactive=None,
		global_inhibition=True, trim=1e-4, seed=None,
		
		# Parameters for the columns
		max_boost=10, duty_cycle=1000,
		
		# Parameters for the segments
		nsynapses=10, seg_th=1,
		
		# Parameters for the synapses
		syn_th=0.5, pinc=0.03, pdec=0.05, pwindow=0.05, 
		random_permanence=True,
		
		# Parameters for fitting (training) the SP
		nepochs=1, clf=None, log_dir=None
		):
		"""
		Initializes this SP Region.
		
		**** The below parameters are for the region ****
		
		@param ninputs: The number of inputs that this region will work with.
		This value should be the same size as the number of inputs in the input
		region.
		
		@param ncolumns: The number of columns to use. The region will be
		created to be 1D with as many columns as specified.
		
		@param pct_active: The percentage of how many columns should be active
		within an inhibition radius. If None, nactive will be used.
		
		@param nactive: An alternative to pct_active, that uses a fixed	number.
		
		@param global_inhibition: Boolean to denote if global inhibition should
		be used.
				
		@param trim: Set to False to disable. If a scalar, any permanence below
		that threshold will be set to 0.
		
		@param seed: The random seed to use for the SP. Set to None to use the
		default seed or provide any valid seed.
		
		**** The below parameters are for the columns ****
		
		@param max_boost: The maximum boost.
		
		@param duty_cycle: The duty cycle to use for boosting. A simple moving
		average will be used based off this many past iterations.
		
		**** The below parameters are for the segments ****
		
		@param nsynapses: The number of synapses that should be connected to
		this segment.
		
		@param seg_th: The number of synapses that must be active for the
		segment to be active.
		
		**** The below parameters are for the synapses ****
		
		@param syn_th: The permanence threshold for when a connection is
		formed. This value should be between 0 and 1, exclusive.
		
		@param pinc: Permanence increment amount.
		
		@param pdec: Permanence decrement amount.
		
		@param pwindow: The offset from "syn_th", which is used to determine
		how the permanence should be initialized. The permanence will have an
		initial value between "syn_th" +/- "pwindow".
		
		@param random_permanence: If this parameter is True, the permanence
		will be randomly selected within the range bound by "pwindow". If this
		parameter is a scalar, the permanence will be fixed to that value. Note
		that clamping automatically occurs, i.e. the permanence is not allowed
		to go above or below 1 or 0, respectively. If None, the permanence will
		be initialized based off the synapse's normalized distance from its
		input. It is highly advised to use the default setting of True. The
		other two options were added for exploratory purposes only. while using
		None better follows the original SP explanation, it will likely reduce
		in a very poor initialization, rendering many synapses useless.
		
		**** Parameters for fitting (training) the SP ****
		
		@param nepochs: The number of epochs to train the SP for.
		
		@param clf: An instance of the classifier to use. The classifier must
		follow the scikit-learn interface. If this parameter is 'None', the SP
		will not be used for classification.
		
		@param log_dir: If not None, various items regarding the SP, including
		the actual instance itself will be saved in the specified directory.
		"""		
		
		# Store the region params
		self.ninputs = ninputs
		self.ncolumns = ncolumns
		self.pct_active = pct_active
		self.nactive = nactive
		self.global_inhibition = global_inhibition
		self.trim = trim
		self.seed = seed
		
		# Store the column params
		self.max_boost = max_boost
		self.duty_cycle = duty_cycle
		
		# Store the segment params
		self.nsynapses = nsynapses
		self.seg_th = seg_th
		
		# Store the synapse params
		self.syn_th = syn_th
		self.pinc = pinc
		self.pdec = pdec
		self.pwindow = pwindow
		self.random_permanence = random_permanence
		
		# Store the training parameters
		self.nepochs = nepochs
		self.clf = clf
		self.log_dir = log_dir
		
		# Initialize the class
		self._init()
	
	def _init(self):
		"""
		Used to initialize the classifier. This is pulled out to ensure that
		the class is compatible with scikit-learn.
		"""
		
		####
		# Directory configuration
		####
		
		# Ensure that only new directories are created. It is assumed that the
		# directory structure is the <run_instance>-<fold_instance>, where
		# <run_instance> is the iteration number for the current set of
		# parameters and <fold_instance> is the iteration number for the
		# current fold.
		if self.log_dir is not None:
			# Test the directory name
			self.log_dir = os.path.abspath(self.log_dir)
			valid_dir = True
			splits = os.path.basename(self.log_dir).split('-')
			if len(splits) != 2:
				valid_dir = False
			else:
				try:
					run_inst, fold_inst = int(splits[0]), int(splits[1])
				except ValueError:
					valid_dir = False
			
			# Build the directory name
			if valid_dir:
				dir = os.path.join(os.path.dirname(self.log_dir),
					'-'.join(s for s in splits[:-1]))
				fmt = '{0}-{{0:0{1}d}}'.format(dir, len(splits[-1]))
				i, dir = 2, fmt.format(fold_inst)
				while os.path.exists(dir):
					dir = fmt.format(i)
					i += 1
				self.log_dir = dir
			else:
				i, dir = 1, self.log_dir
				while os.path.exists(self.log_dir):
					self.log_dir = '{0}-{1}'.format(dir, i)
					i += 1
			
			# Create the directory
			os.makedirs(self.log_dir)
			
			# Save the parameters
			params = self.get_params()
			if self.clf is not None:
				params['clf'] = str(params['clf']).replace('\n    ', '')
			s = json.dumps(params, sort_keys=True, indent=4,
				separators=(',', ': ')).replace('},', '},\n')
			with open(os.path.join(self.log_dir, 'config.json'), 'wb') as f:
				f.write(s)
		
		####
		# Region configuration
		####
		
		self.inhibition_radius = self.ncolumns # Adjusted later if necessary
		self.learn = True # Initialize learning
		self.x = None # The current sample
		
		# Keep a random number generator internally to ensure the global state
		# is unaltered
		self.prng = np.random.RandomState()
		self.prng.seed(self.seed)
		
		####
		# Column configuration
		####
		
		# Prepare the histories
		self.overlap = np.zeros((self.ncolumns, self.duty_cycle), dtype='i')
		self.y = np.zeros((self.ncolumns, self.duty_cycle), dtype='bool')
		
		# Prepare the boosts
		self.boost = np.ones(self.ncolumns)
		
		# Prepare the duty cycles
		self.active_dc = np.zeros(self.ncolumns)
		self.overlap_dc = np.zeros(self.ncolumns)
		
		# Prepare the neighborhood
		self.neighbors = np.zeros((self.ncolumns, self.ncolumns), dtype='bool')
		
		####
		# Segment configuration
		####
		
		# Mapping of synapse to lower region
		#   - Random connectivity to anywhere in the region
		#   - Note: np.random.permutation is faster than np.random.choice
		self.syn_map = np.zeros((self.ncolumns, self.nsynapses), dtype='i')
		s = np.arange(self.ninputs)
		for i in xrange(self.ncolumns):
			self.syn_map[i] = self.prng.permutation(s)[:self.nsynapses]
		
		# Compute the distances between all synapses and their inputs
		#   - The distance is the Euclidean distance
		#   - This distance matrix is unstable, as it assumes inputs and
		#     columns coincide if they have the same index. Due to how this
		#     matrix is utilized this does not impact the actual calculation.
		#     The instability was allowed to allow the distance calculation to
		#     be 1D.
		#   - This value is cached for local inhibition and deleted after
		#     being used for permanence initialization for global inhibition.
		self.syn_dist = np.zeros((self.ncolumns, self.nsynapses), dtype='i')
		for i, dest in enumerate(self.syn_map):
			self.syn_dist[i] = np.abs(dest - i, dtype='i')
		
		####
		# Synapse configuration
		####
		
		# Initialization of the permanences
		if self.random_permanence is None:
			# - Permanences are set to be close to be the connected threshold
			# - Permanences follow a uniform distribution based off the
			#   distance from the input to the column. The closer the distance
			#   the larger the permanence.
			#     - To ensure fairness, the distances are normalized based off
			#       the max distance that an input can be from each column
			self.p = np.zeros((self.ncolumns, self.nsynapses))
			pmin = float(max(self.syn_th - self.pwindow, 0))
			pmax = float(min(self.syn_th + self.pwindow, 1))
			pwin = 2 * self.pwindow
			for i, syn_d in enumerate(self.syn_dist):
				self.p[i] = pmin + pwin * (1 - (syn_d /
					float(max(self.ncolumns - i, i))))
		elif self.random_permanence is True:
			# - Randomly initialize the permanence via a uniform distribution
			#   within the bounds of the window
			pmin = float(max(self.syn_th - self.pwindow, 0))
			pmax = float(min(self.syn_th + self.pwindow, 1))
			self.p = self.prng.uniform(pmin, pmax, (self.ncolumns,
				self.nsynapses))
		else:
			# - Fix the permanence the scalar value
			self.p = np.zeros((self.ncolumns, self.nsynapses))
			self.p.fill(self.random_permanence)
		
		# Initialize the connected synapse mask
		self.syn_c = self.p >= self.syn_th
		
		# Perform some cleanup
		if self.global_inhibition: del self.syn_dist
		
		####
		# Fitting parameters
		####
		
		# The SP's predictions on the training data. Updated by calling "fit".
		# This is the traditional output of the SP.
		self.column_activations = None
		
		# The dataset's training "y" values. Updated by calling "fit".
		# Note: This is only used if classification is being performed.
		self.tr_y = None
		
		# The probabilities for input importance. Updated by calling
		# "get_probabilities".
		self.prob = None
		
		####
		# Constants
		####
		
		# Factor used in updating the permanences
		self.c_pupdate = self.pinc + self.pdec
		
		# Factor using in determining the minimum duty cycles
		self.c_mdc = 0.01
		
		# Factor used in determining the column boost amount
		self.c_cboost = 1. - self.max_boost
		
		# Amount to boost the synapses' permanence by
		self.c_sboost = 0.1 * self.syn_th
	
	def get_params(self, deep=True):
		"""
		Get the parameters for this class.
		
		@param deep: This parameter is completely ignored. It is here to
		satisfy the interface requirements
		
		@return: The parameters for this object.
		"""
		
		return {'ninputs':self.ninputs, 'ncolumns':self.ncolumns,
			'pct_active':self.pct_active, 'nactive':self.nactive,
			'global_inhibition':self.global_inhibition, 'trim':self.trim,
			'seed':self.seed, 'max_boost':self.max_boost,
			'duty_cycle':self.duty_cycle, 'nsynapses':self.nsynapses,
			'seg_th':self.seg_th, 'syn_th':self.syn_th, 'pinc':self.pinc,
			'pdec':self.pdec, 'pwindow':self.pwindow,
			'random_permanence':self.random_permanence, 'nepochs':self.nepochs,
			'clf':self.clf, 'log_dir':self.log_dir}
	
	def set_params(self, **parameters):
		"""
		Set the parameters for this class.
		
		@param parameters: All of the parameters to be set. Refer to the class'
		__init__ method for more details.
		
		@return: An instance to this object.
		"""
		
		# Update the parameters
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		
		# Reinitialize
		self._init()
		
		return self
	
	def _get_num_cols(self):
		"""
		Get the number of k-columns to select within the neighborhood.
		"""
		
		if self.nactive is None:
			return max(int(self.pct_active * self.inhibition_radius), 1)
		return self.nactive
	
	def _update_inhibition_radius(self):
		"""
		Sets the inhibition radius based off the average receptive field size.
		The average receptive field size is the distance of the connected
		synapses with respect to to their input column. In other words, it is
		the distance between a column and its input source averaged across all
		connected synapses. The distance used is the Euclidean distance. Refer
		to the initialization of self.syn_dist for more details.
		
		NOTE
			- This should only be called after phase 1.
			- The minimum inhibition radius is lower-bounded by 1.		
		"""
		
		self.inhibition_radius = max(bn.nansum(self.syn_dist * self.syn_c) /
			max(bn.nansum(self.syn_c), 1), 1)
	
	def _compute_neighborhood(self):
		"""
		Compute the neighborhood for each column as a bit mask. 1D is being
		used, so neighbors are selected along a number line with the ends
		clamped.
		
		NOTE - This should be called after updating the inhibition radius.
		"""
		
		for i in xrange(self.ncolumns):
			self.neighbors[i][np.arange(max(0, i - self.inhibition_radius),
				min(i + self.inhibition_radius + 1, self.ncolumns))] = 1
	
	def _update_active_duty_cycle(self):
		"""
		Update the active duty cycle.
		"""
		
		self.active_dc = np.mean(self.y, 1)
	
	def _update_overlap_duty_cycle(self):
		"""
		Update the overlap duty cycle.
		"""
		
		self.overlap_dc = np.mean(self.overlap >= self.seg_th, 1)
	
	def _update_boost(self, min_dc):
		"""
		Update the boost. Boost linearly decreases with respect to the active
		duty cycle. If the active duty cycle is greater than the minimum duty
		cycle the boost is disabled (set to 1).
		
		@param min_dc: The minimum duty cycles.
		"""
		
		for i, (mdc, adc) in enumerate(izip(min_dc, self.active_dc)):
			if mdc == 0:
				self.boost[i] = self.max_boost
			elif adc > mdc:
				self.boost[i] = 1
			else:
				self.boost[i] = (self.c_cboost / mdc) * adc + self.max_boost
	
	def _phase1(self):
		"""
		Execute phase 1 of the SP region. This phase is used to compute the
		overlap.
		
		Note - This should only be called once the input has been updated.
		"""
		
		# Compute the connected synapse mask
		self.syn_c = self.p >= self.syn_th
		
		# Compute the overlaps
		self.overlap[:, 1:] = self.overlap[:, :-1] # Shift
		self.overlap[:, 0] = bn.nansum(self.x[self.syn_map] * self.syn_c, 1)
		self.overlap[:, 0][self.overlap[:, 0] < self.seg_th] = 0
		self.overlap[:, 0] = self.overlap[:, 0] * self.boost
	
	def _phase2(self):
		"""
		Execute phase 2 of the SP region. This phase is used to compute the
		active columns.
		
		Note - This should only be called after phase 1 has been called and
		after the inhibition radius and neighborhood have been updated.
		"""
		
		# Shift the outputs
		self.y[:, 1:] = self.y[:, :-1]
		self.y[:, 0] = 0
		
		# Calculate k
		#   - For a column to be active its overlap must be above the overlap
		#     value of the k-th largest column in its neighborhood.
		k = self._get_num_cols()
		
		if self.global_inhibition:
			# The neighborhood is all columns, thus the set of active columns
			# is simply columns that have an overlap above the k-th largest
			# in the entire region
			
			# Compute the winning column indexes
			if self.learn:				
				# Randomly break ties
				ix = bn.argpartsort(-self.overlap[:, 0] -
					self.prng.uniform(.1, .2, self.ncolumns), k)[:k]
			else:
				# Choose the same set of columns each time
				ix = bn.argpartsort(-self.overlap[:, 0], k)[:k]
			
			# Set the active columns
			self.y[ix, 0] = self.overlap[ix, 0] > 0
		else:
			# The neighborhood is bounded by the inhibition radius, therefore
			# each column's neighborhood must be considered
			
			for i in xrange(self.ncolumns):
				# Get the neighbors
				ix = np.where(self.neighbors[i])[0]
				
				# Compute the minimum top overlap
				if ix.shape[0] <= k:
					# Desired number of candidates is at or below the desired
					# activity level, so find the overall max
					m = max(bn.nanmax(self.overlap[ix, 0]), 1)
				else:
					# Desired number of candidates is above the desired
					# activity level, so find the k-th largest
					m = max(bn.argpartsort(-self.overlap[ix, 0], k)[k], 1)
				
				# Set the column activity
				if self.overlap[i, 0] >= m: self.y[i, 0] = True
	
	def _phase3(self):
		"""
		Execute phase 3 of the SP region. This phase is used to conduct
		learning.
		
		Note - This should only be called after phase 2 has been called.
		"""
		
		# Notes:
		# 1. logical_not is faster than invert
		# 2. Multiplication is faster than bitwise_and which is faster than
		#    logical_not
		# 3. Slightly different format than original definition
		#    (in the comment) to get even more speed benefits
		"""
		x = self.x[self.syn_map]
		self.p = np.clip(self.p + self.y[:, 0:1] * (x * self.pinc -
			np.logical_not(x) * self.pdec), 0, 1)
		"""
		self.p = np.clip(self.p + (self.c_pupdate * self.y[:, 0:1] *
			self.x[self.syn_map] - self.pdec * self.y[:, 0:1]), 0, 1)
		
		# Update the boosting mechanisms
		if self.global_inhibition:
			min_dc = np.zeros(self.ncolumns)
			min_dc.fill(self.c_mdc * bn.nanmax(self.active_dc))
		else:
			min_dc = self.c_mdc * bn.nanmax(self.neighbors * self.active_dc, 1)
		self._update_active_duty_cycle()
		self._update_boost(min_dc)
		self._update_overlap_duty_cycle()
		
		# Boost permanences
		mask = self.overlap_dc < min_dc
		mask.resize(self.ncolumns, 1)
		self.p = np.clip(self.p + self.c_sboost * mask, 0, 1)
		
		# Trim synapses
		if self.trim is not False:
			self.p[self.p < self.trim] = 0

	def step(self, x):
		"""
		Step the SP for the provided input.
		
		@param x: The input to work with. This should be a NumPy array with
		the same shape as ninputs.
		"""
		
		# Update the current input and execute phase 1
		self.x = np.array(x, dtype='bool') # Input must be binary!
		self._phase1()
		
		# Update the inhibition radius and the neighbors and execute phase 2
		if not self.global_inhibition:
			self._update_inhibition_radius()
			self._compute_neighborhood()
		self._phase2()
		
		# Execute phase 3
		if self.learn: self._phase3()
	
	def execute(self, x, store=True):
		"""
		Step the SP for each xi in x.
		
		@param x: The input to work with. This should be a 2D NumPy array with
		the shape as (nsamples, ninputs).
		
		@param store: If True the predictions are stored.
		
		@return: The predictions if store is True; otherwise, None.
		"""
		
		# Reshape x if needed
		ravel = False
		if len(x.shape) == 1:
			ravel = True
			x = x.reshape(1, x.shape[0])
		
		# Execute the SP for each input, saving the results
		if store:
			prediction = np.array(np.zeros((x.shape[0], self.ncolumns)),
				dtype='bool')
			for i, xi in enumerate(x):
				self.step(xi)
				prediction[i] = self.y[:, 0]
			
			return prediction if not ravel else prediction.ravel()
		
		# Execute the SP for each input, ignoring the results
		for i, xi in enumerate(x):
			self.step(xi)
	
	def fit(self, x, y=None):
		"""
		Fit the SP on the data.
		
		@param x: The input to work with. This should be a NumPy array with
		the same shape as ninputs.
		
		@param y: This parameter is only used for fitting the classifier that
		the SP will be using.
		"""
		
		# Start the fit timer
		tg = time.time()
		
		# Train for the specified number of epochs
		self.learn = True
		for _ in xrange(self.nepochs):
			self.execute(x, False)
		
		# Time for learning, only
		tl = time.time() - tg
		
		# Time for predicting
		tp = time.time()
		
		# Get the training results
		self.learn = False
		self.column_activations = self.execute(x)
		
		# Stop the timers
		t = time.time()
		tg = t - tg
		tp = t - tp
		
		# Log the results if desired
		if self.log_dir is not None:
			self._save_data('column_activations-train',
				self.column_activations)
			self._save_data('permanence', self.p)
			self._save_data('synaptic_connectivity_map', self.syn_map)
			self.save('sp')
			self._log_stats('Total Fitting Time [s]', tg)
			self._log_stats('Learning Training Time [s]', tl)
			self._log_stats('Prediction Training Time [s]', tp)
		
		# Store the training Y values
		if self.clf is not None: self.tr_y = y
	
	def predict(self, x):
		"""
		Get the prediction results.
		
		@param x: The data to predict. If this is 2D it is assumed that
		multiple predictions will be made; otherwise, a single prediction will
		be made.
		
		@return: The predicted SDR.
		"""
		
		self.learn = False
		return self.execute(x)
	
	def score(self, x, y, tr_x=None, tr_y=None, score_method='column'):
		"""
		Compute the accuracy for y. Note that if no classifier was specified,
		this function will return None. This method must be called after
		fitting the SP.
		
		@param x: The data to predict. If this is 2D it is assumed that
		multiple predictions will be made; otherwise, a single prediction will
		be made.
		
		@param y: The expected results. This should be the expected result from
		the original data, not the expected output of the SP.
		
		@param tr_x: The original data. This should be the same data that was
		used to train the SP.
		
		@param tr_y: The training labels. Set to None to use the stored labels.
		
		@param score_method: The method to use for computing the accuracy. This
		must be one the following: "column" - Uses the set of active columns
		to compute the accuracy, "prob" - Use the probabilistic version of the
		input, or "reduction" - Uses the dimensionality reduced version of the
		input. To use any method other than "column", tr_x must be provided.
		
		@return: The accuracy or None if either fitting was not performed or
		no classifier was provided.
		"""
		
		# Check the internal state
		if tr_y is None: tr_y = self.tr_y
		if (self.clf is None) or (self.column_activations is None) or \
			(tr_y is None): return None
		if tr_x is None: score_method = 'column'
		
		# Start the predict timer
		t = time.time()
		
		# Prep the data for fitting
		if score_method == 'prob':
			tr_x = self.transform(tr_x)
			te_x = self.transform(x)
		elif score_method == 'reduction':
			tr_x = self.reduce_dimensions(tr_x)
			te_x = self.reduce_dimensions(x)
		else: # Default to 'column'
			score_method = 'column' # Set again to be safe
			tr_x = self.column_activations
			te_x = self.predict(x)
		
		# Stop the predict timer
		t = time.time() - t
		
		# Fit the classifier on the SP's training prediction
		self.clf.fit(tr_x, tr_y)
		score = self.clf.score(te_x, y)
		
		# Log the results if desired
		if self.log_dir is not None:
			if score_method == 'prob':
				self._save_data('probabilistic-train', tr_x)
				self._save_data('probabilistic-test', te_x)
			elif score_method == 'reduction':
				self._save_data('reduction-train', tr_x)
				self._save_data('reduction-test', te_x)
			else:
				self._save_data('column_activations-test', te_x)
			self._log_stats('Predicting Testing Time: {0} [s]'.format(
				score_method), t)
			self._log_stats('Accuracy: {0}'.format(score_method), score)
		
		return score
	
	def get_probabilities(self, store=True):
		"""
		Get the probabilities associated with each feature. This technique
		uses the max across probabilities to form the global probabilities.
		This method should be called after fitting the SP.
		
		@param store: If True, the probabilities are stored internally. Set to
		False to reduce memory.
		
		@return: Return the probabilities.
		"""
		
		# Get the probabilities
		prob = np.zeros(self.ninputs)
		for i in xrange(self.ninputs):
			# Find all of the potential synapses for this input
			valid = self.syn_map == i
			
			# Find the max permanence across each of the potential synapses
			try:
				prob[i] = bn.nanmax(self.p[valid])
			except ValueError:
				prob[i] = 0. # Occurs for missing connections
		
		# Store the probabilities
		if store: self.prob = prob
		
		return prob
	
	def transform(self, x):
		"""
		Transform each value in x to be a probability. This method should be
		called after fitting the SP.
		
		@param x: The data to reduce. If it is a vector, it is assumed to be
		one sample containing the features. If it is a matrix, it is assumed
		that each row represents a sample.
		
		@return: The probabilistic version of x.
		"""
		
		# Check to see if the probabilities have been computed
		if self.prob is None: self.get_probabilities()
		
		# Transform x
		x2 = np.zeros(x.shape)
		for i, xi in enumerate(x):
			x2[i] = self.prob * xi
		
		return x2
	
	def reduce_dimensions(self, x):
		"""
		Apply the dimensionality reduction on x. This method should be called
		after fitting the SP.
		
		@param x: The data to reduce. If it is a vector, it is assumed to be
		one sample containing the features. If it is a matrix, it is assumed
		that each row represents a sample.
		
		@return: The transformed version of x.
		"""
		
		# Check to see if the probabilities have been computed
		if self.prob is None: self.get_probabilities()
		
		# Reshape x if needed
		ravel = False
		if len(x.shape) == 1:
			ravel = True
			x = x.reshape(1, x.shape[0])
		
		# Determine which dimensions are valid
		valid = np.where(self.prob >= self.syn_th)[0]
		
		# Reduce x
		x2 = np.zeros((len(x), len(valid)))
		for i, xi in enumerate(x):
			x2[i] = xi[valid]
		
		return x2 if not ravel else x2.ravel()
	
	def reconstruct_input(self, x=None):
		"""
		Reconstruct the original input using only the stored permanences and
		the set of active columns. The maximization of probabilities approach
		is used. This method must be called after fitting the SP.
		
		@param x: The set of active columns or None if the SP was never fitted.
		"""
		
		# Check input
		if x is None: x = self.column_activations
		if x is None: return None
		
		# Reshape x if needed
		ravel = False
		if len(x.shape) == 1:
			ravel = True
			x = x.reshape(1, x.shape[0])
		
		# Get the input mapping
		imap = [np.where(self.syn_map == i) for i in xrange(self.ninputs)]
		
		# Get the reconstruction
		x2 = np.zeros((x.shape[0], self.ninputs))
		for i, xi in enumerate(x):
			# Mask off permanences not relevant to this input
			y = self.p * xi.reshape(self.ncolumns, 1)
			
			# Remap permanences to input domain
			for j in xrange(self.ninputs):
				# Get the max probability across the current input space
				try:
					x2[i][j] = bn.nanmax(y[imap[j]])
				except ValueError:
					x2[i][j] = 0. # Occurs for missing connections
				
				# Threshold back to {0, 1}
				x2[i][j] = 1 if x2[i][j] >= self.syn_th else 0
		
		return x2 if not ravel else x2.ravel()