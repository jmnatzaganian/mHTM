# parallel.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 12/08/15
# 
# Description    : Module for parallel processing.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Module for parallel processing.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import subprocess, time, os
from itertools import product, izip

# Third party imports
from scipy.stats import randint

# Program imports
from mHTM import prob

runner_template = (
	#: The runner template to use. This is a BASH script that allows you to run
	#: your jobs.
	#:
	#: It must have the following items:
	#:	- {0} = The name of the job
	#:	- {1} = The full path to the standard out file
	#:	- {2} = The full path to the standard error file
	#:  - {3} = The name of the partition to use
	#:  - {4} = The maximum time limit.
	#:  - {5} = The maximum memory requirements in MB.
	#:  - {6} = The command to execute. This should be the full path to the
	#:  file that is being executed along with any command line arguments.
	
	'#!/bin/bash -l\n'
	'# Name of the job - Must be unique\n'
	'#SBATCH -J {0}\n\n'
	
	'# Standard out and Standard Error output files\n'
	'#SBATCH -o {1}\n'
	'#SBATCH -e {2}\n\n'
	
	'# Set the partition and the number of cores to use\n'
	'#SBATCH -p {3} -n 1\n\n'
	
	'# Set the maximum time limit\n'
	'#SBATCH -t {4}\n\n'
	
	'# Set the memory requirements in MB\n'
	'#SBATCH --mem={5}\n\n'
	
	'# Run command\n'
	'{6}'
	)

def create_runner(command, runner_path, template=runner_template,
	job_name='job', stdio_path='stdio.txt', stderr_path='stderr.txt',
	partition_name='debug', time_limit='00-06:00:00', memory_limit=512):
	"""
	Create a runner (a BASH script) to run the specified job.
	
	@param command: The command to execute. This should be the full path to the
	file that is being executed along with any command line arguments.
	
	@param runner_path: The full path to create the runner.
	
	@param template: The template BASH script to use. Refer to the
	included template, "runner_template", for more details.
	
	@param job_name: The name of the job (should be unique).
	
	@param stdio_path: The full path to the standard out file.
	
	@param stderr_path: The full path to the standard error file.
	
	@param partition_name: The name of the partition.
	
	@param time_limit: The maximum time limit.
	
	@param memory_limit: The maximum memory requirements in MB.
	
	@return: A string containing the new BASH script.
	"""
	
	# Make the runner
	runner = template.format(job_name, stdio_path, stderr_path, partition_name,
		time_limit, memory_limit, command)
	
	# Create the file
	with open(runner_path, 'wb') as f:
		f.write(runner)

def execute_runner(runner_path, queue_limit=5000):
	"""
	Execute the runner script.
	
	@param runner_path: The full path to create the runner.
	
	@param queue_limit: The max number of jobs to have in the queue. If more
	jobs are being launched than are able to be handled by the queue, the
	system will sleep, and then try again.
	"""
	
	def queue_free():
		"""
		Checks the queue to see if there is room to add more jobs.
		
		@return: True if there is room, else False.
		"""
		
		awk_cmd  = '$6=="0:00" {++count} END {print count-1}'
		queue    = subprocess.Popen('squeue', stdout=subprocess.PIPE)
		num_jobs = int(subprocess.check_output(('awk', awk_cmd),
			stdin=queue.stdout).strip())
		
		return num_jobs < queue_limit
	
	# Wait until there is enough room in the queue
	while not queue_free():
		print 'Queue full - Sleeping until it is free....'
		time.sleep(600)
	
	# Execute file
	while True:
		p = subprocess.Popen(('sbatch', runner_path), stdout=subprocess.PIPE,
			stderr=subprocess.PIPE)
		error = p.stderr.read()
		if len(error) == 0:
			break
		else:
			print error
			print 'Sleeping and retrying!'
			time.sleep(10)

class ParamGenerator(object):
	"""
	Build parameters properly bounded for the SP. Item generation follows
	scikit-learn's ParameterSampler; therefore, the order is guaranteed.
	
	Assumes that at least the following parameters are being varied: ncolumns,
	nactive, nsynapses, seg_th, and log_dir.
	"""
	
	def __init__(self, param_distributions, niter, nsplits, ninputs):
		"""
		Initialize this class.
		
		@param param_distributions: Dictionary containing the items to vary.
		
		@param niter: The number of iterations to perform.
		
		@param nsplits: The number of splits of the data.
		
		@param ninputs: The number of inputs to the SP.
		"""
		
		# Store the parameters
		self.param_distributions = param_distributions
		self.niter = niter
		self.nsplits = nsplits
		self.ninputs = ninputs
		
		# Order to yield items
		self.keys = sorted(self.param_distributions.keys())
		
		# Build the generator
		self.gen = self.make_data()
	
	def rvs(self):
		"""
		Return the next item from the generator.
		
		@return: The next parameter value.
		"""
		
		return self.gen.next()
	
	def make_data(self):
		"""
		Create a generator for building the data.
		"""
		
		# Build a logging format suitable for sorting
		log_dir = self.param_distributions['log_dir']
		log_fmt = '{{0:0{0}d}}-{{1:0{1}d}}'.format(len(str(self.niter)),
			len(str(self.nsplits)))
		
		# Create the parameters for each instance
		for i in xrange(1, self.niter + 1):
			# Make the params
			ncolumns = self.param_distributions['ncolumns'].rvs()
			param = {'ncolumns':ncolumns}
			
			####
			# Ensure that parameters make sense
			####
			
			# Compute nactive as a function of ncolumns
			# Ensure that nactive is bounded by (0, ncolumns)
			nactive = int(self.param_distributions['nactive'].rvs() * ncolumns)
			while ((nactive == 0) or (nactive == ncolumns)):
				nactive = int(self.param_distributions['nactive'].rvs() *
					ncolumns)
			param['nactive'] = nactive
			
			# Ensure that each input is seen at least once
			nsynapses = self.param_distributions['nsynapses'].rvs()
			p = prob.p_c(ncolumns, nsynapses, self.ninputs)
			e = int(p * self.ninputs)
			while e > 0:
				nsynapses = self.param_distributions['nsynapses'].rvs()
				p = prob.p_c(ncolumns, nsynapses, self.ninputs)
				e = int(p * self.ninputs)
			param['nsynapses'] = nsynapses
			
			# Compute seg_th as a function of nsynapses
			seg_th = int(self.param_distributions['seg_th'].rvs() * nsynapses)
			param['seg_th'] = seg_th
			
			# Make a useful log directory
			param['log_dir'] = os.path.join(log_dir, log_fmt.format(i, 1))
			
			####
			# Add all other parameters
			####
			
			added = set(param.keys())
			missing = [key for key in self.keys if key not in added]
			for key in missing:
				if hasattr(self.param_distributions[key], 'rvs'):
					param[key] = self.param_distributions[key].rvs()
				else:
					param[key] = value[randint(0, len(v)).rvs()]
			
			# Yield each item
			for key in self.keys:
				yield param[key]

class ConfigGenerator(object):
	"""
	Build parameter configurations. This class creates full dictionary
	configuration objects for the provided parameters.
	
	This class assumes that either iterables or static items are provided for
	each parameter. In the case of multiple iterables all possible parameter
	combinations are generated.
	
	Example:
	base params: {'a':[0,1], 'b':[2,3]}
	parameter sets: (a=0, b=2), (a=0, b=3), (a=1, b=2), (a=1, b=3)
	"""
	
	def __init__(self, base_config, ntrials=1):
		"""
		Initialize this class.
		
		@param base_config: Dictionary containing the base configuration. All
		desired parameters should be included in this dictionary.
		
		@param ntrials: The number of trials to perform for each configuration.
		Each trial may use a different set of input data or be a different
		random initialization. In the event that both are being used, provide
		the product of those two numbers.
		"""
		
		self.base_config = base_config
		self.ntrials = ntrials
	
	def get_config(self):
		"""
		Create a generator for building the data.
		"""
		
		# Split out iterables and non iterables
		static_params, dynamic_params = {}, {}
		for key, value in self.base_config.items():
			if hasattr(value, '__iter__'):
				dynamic_params[key] = list(value)
			else:
				if key != 'log_dir':
					static_params[key] = value
		
		# Build a logging format suitable for sorting
		try:
			log_dir = self.base_config['log_dir']
		except KeyError:
			log_dir = None
		if log_dir is not None:
			# Compute the number of configurations
			nconfigs = 0
			for i in enumerate(product(*dynamic_params.values())):
				nconfigs += 1
			
			# Make the log format
			log_fmt = os.path.join(log_dir,
				'{{0:0{0}d}}-'.format(len(str(nconfigs))) +
				'{{0:0{0}d}}'.format(len(str(self.ntrials))).format(1))
		
		# Create all possible set of dynamic parameters
		for i, items in enumerate(product(*dynamic_params.values()), 1):
			config = static_params.copy()
			
			# Add all dynamic items
			for item, param in izip(items, dynamic_params): 
				config[param] = item
			
			# Make a useful log directory
			if log_dir is not None: config['log_dir'] = log_fmt.format(i, 1)
			
			yield config
