# util.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 09/27/14
#	
# Description    : Utility module.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Utility module. This module handles any sort of accessory items.

G{packagetree nano_htm}
"""

__docformat__ = 'epytext'

# Native imports
import os, time, random, json, csv
from uuid import uuid4

# Third party imports
import numpy as np

# Program imports
from errors import InvalidSequence

###############################################################################
########## Primary Functions
###############################################################################

def get_shape(seq):
	"""
	Returns a list containing the dimensions of an N-Dimensional sequence.
	This function assumes that the size of a dimension is the same for that
	level in the sequence. Additionally, this function does everything by
	reference, so no accessory temporary memory will be dealt with. The
	original list is also left intact.
	
	@param seq: A sequence of 0 or more elements.
	
	@raise InvalidSequence: Raised if "seq" is not a sequence.
	"""
	
	# Initializations
	l     = seq
	shape = []
	
	# Make sure it is a valid sequence for this function
	try:
		len(l)
	except:
		raise InvalidSequence(seq, "len")
	
	# Recursively determine the shape
	while hasattr(l, '__iter__'):
		shape.append(len(l))
		if len(l) == 0:
			break
		l = l[0] # Update list reference
	
	return shape

def flatten(seq):
	"""
	Flattens a sequence of elements. Creates a generator.
	
	@param seq: The sequence to flatten.
	
	@return: Generator of the flattened sequence.
	"""
	
	if not hasattr(seq, '__iter__'):
		yield seq
	else:
		for item in seq:
			if hasattr(item, '__iter__'):
				for sub_item in flatten(item):
					yield sub_item
			else:
				yield item

def get_1d_pos(position, shape=[[0, 27], [0, 27]]):
	"""
	Convert a 2D representation to 1D.
	
	@param position: The 2D position (sequence).
	
	@param shape: The shape of the region.
	"""
	
	return position[1] + position[0] * (shape[1][1] - shape[1][0] + 1)

def get_2d_pos(position, shape=[[0, 27], [0, 27]]):
	"""
	Convert a 1D representation to 2D.
	
	@param position: The 1D position (sequence).
	
	@param shape: The shape of the region.
	"""
	
	y_len = shape[1][1] - shape[1][0] + 1
	x     = position % y_len
	y     = (position - x) / y_len
	
	return [x, y]

def timestamp():
	"""
	Generates a timestamp.
	
	@return: The current timestamp.
	"""
	
	return time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

def mkudir(base_dir, base_name=None):
	"""
	Creates a unique directory.
	
	@param base_dir: The full path to the base directory.
	
	@param base_name: The base name to use for the directory. If the path with
	this name does not exist, this will become the path; otherwise, a
	timestamp will be appended to it. Set to None to ignore the base_name.
	
	@return: The full path to the directory that was created.
	"""
	
	# Check for simple case
	if base_name is not None:
		sub_dir = os.path.join(base_dir, base_name)
		if not os.path.isdir(sub_dir):
			os.makedirs(sub_dir)
			return sub_dir
	
	# More advanced case
	try:
		if base_name is not None:
			sub_dir = os.path.join(base_dir, '{0}-{1}'.format(base_name,
				timestamp()))
		else:
			sub_dir = os.path.join(base_dir, '{0}'.format(timestamp()))
		os.makedirs(sub_dir)
		return sub_dir
	except:
		time.sleep(random.randint(1, 3)) # Offset the current directory name
		return mkudir(base_dir)

def mkdir(path):
	"""
	Make a new directory.
	
	@param path: The full path to the directory to create.
	"""
	
	if not os.path.isdir(path):
		os.makedirs(path)

def rmse(actual, predicted):
	"""
	Compute the root mean squared error.
	
	@param actual: The expected output.
	
	@param predicted: The obtained output.
	
	@return: The RMSE.
	"""
	
	# Remove None values
	p, a = [], []
	for i, x in enumerate(predicted):
		if x is not None:
			p.append(x)
			a.append(actual[i])
	
	return np.nanmean((np.array(a) - np.array(p)) ** 2) ** 0.5

def create_unique_name():
	"""
	Generate a random (likely) unique name.
	
	@return: A string representing the name.
	"""
	
	return ''.join(str(uuid4()).split('-'))