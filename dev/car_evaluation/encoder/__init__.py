# __init__.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 12/20/14
#	
# Description    : Defines the encoder package
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
This package is used to encode raw input data into an HTM friendly format. It
supports various formats. The chosen encoder may simply be a unity function,
such that all other operations have been completed, or it may be the full
pre-processing suite. If heavy pre-processing is required, it is advised to do
those steps ahead of time and use the unity encoder.

When encoding data, each input parameter will be used to represent the state of
a column. If the column's state is 1 it is active, otherwise it is inactive.
Synapses will be bound to these columns in a one-to-many relationship. Use this
knowledge very carefully when deciding how to encode your data.

The following shows the modules contained in this package:

G{packagetree nano_htm}
"""

__docformat__ = 'epytext'

# Program imports
from base import Encoder
from scalar import Scalar
from multi import Multi
from category import Category
from errors import BaseException, wrap_error

###############################################################################
########## Exception Handling
###############################################################################

class UnsupportedEncoder(BaseException):
	"""
	Exception if the specified encoder is not supported.
	"""
	
	def __init__(self, name):
		"""
		Initialize this class.
		
		@param name: The name of the encoder.
		"""
		
		self.msg = wrap_error('The desired encoder, {0}, is not supported. '
			'Please request a valid encoder.'.format(name))

###############################################################################
########## Functions
###############################################################################

def get_encoder(type, **kargs):
	"""
	Creates an encoder of the appropriate type and returns an instance of it.
	
	@param type: The type of encoder to create. The supported types are:
	"unity", "threshold", "scalar", "category", and "multi".
	
	@param kargs: Any keyword arguments to pass to the encoder.
	
	@return: An encoder instance.
	
	@raise UnsupportedEncoder: Raised if the requested encoder is not
	supported.
	"""
	
	
	t = type.lower()
	
	if t == 'unity':
		return Unity(**kargs)
	elif t == 'threshold':
		return Threshold(**kargs)
	elif t == 'scalar':
		return Scalar(**kargs)
	elif t == 'multi':
		return Multi(**kargs)
	elif t == 'category':
		return Category(**kargs)
	else:
		raise UnsupportedEncoder(t)

def is_finite(encoder):
	"""
	Determine if the encoder has a finite number of bins. Technically all
	encoders do, but this refers to those that have a feasibly finite number
	of bins. For example, a scalar encoder with 100 bins has a small number of
	bins (100), thus it is finite. A unity encoder with only 10 bits has a
	large number of bins (10!), thus it is not finite.
	
	@param encoder: An encoder instance.
	"""
	
	t = type(encoder).__name__.lower()
	
	if t == 'scalar' or t == 'category':
		return True
	return False