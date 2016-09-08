# category.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 08/05/14
#	
# Description    : Implementation of a category encoder
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Implementation of a category encoder. A multi-hot encoding of each category is
used.

G{packagetree nano_htm}
"""

__docformat__ = 'epytext'

# Third party imports
import numpy as np

# Program imports
from encoder import Encoder, Scalar
from errors import BaseException, wrap_error

###############################################################################
########## Exception Handling
###############################################################################

class InvalidCategoryEncoderNumberCategories(BaseException):
	"""
	Exception if the number of categories is too small.
	"""
	
	def __init__(self, ncategories):
		"""
		Initialize this class.
		
		@param ncategories: The number of categories.
		"""
		
		self.msg = wrap_error('The requested number of categories, {0}, is too'
			' low. There must be at least one category.'.format(ncategories))

class InvalidCategoryEncoderNumberBits(BaseException):
	"""
	Exception if the number of bits is too small.
	"""
	
	def __init__(self, nbits, ncategories):
		"""
		Initialize this class.
		
		@param nbits: The number of bits to use.
		
		@param ncategories: The number of categories.
		"""
		
		self.msg = wrap_error('The requested number of bits, {0}, is too'
			' low. There must be at least as many bits as the requested number'
			' of categories, {1}.'.format(nbits, ncategories))

class InvalidCategoryEncoderCategory(BaseException):
	"""
	Exception if an unknown category is being encoded.
	"""
	
	def __init__(self, desired, valid):
		"""
		Initialize this class.
		
		@param desired: The desired category to encode.
		
		@param valid: The valid categories.
		"""
		
		self.msg = wrap_error('The category, {0}, is not valid. Valid '
			'categories are: {1}.'.format(desired, ', '.join(str(c) for c in
			valid)))

###############################################################################
########## Class Implementations
###############################################################################

class Category(Encoder):
	"""
	Category encoder. This encoder uses a multi-hot encoding for each category.
	All categories are treated as integers; therefore, when encoding a single
	integer should be provided that is in the inclusive set
	{0, ..., num_categories - 1}.
	"""
	
	def __init__(self, num_categories, active_bits=None, num_bits=None,
		**kargs):
		"""
		Initializes this encoder.
		
		@param num_categories: The number of categories the encoder will have.
		
		@param active_bits: The number of active bits in the number (bits that
		are '1'). If None, num_bits must be provided.
		
		@param num_bits: The number of bits to use. If None, this will be sized
		to fit. If not None, the active_bits will be scaled as needed.
		
		@param kargs: Any keyword arguments to be passed to the base class.
		
		@raise InvalidCategoryEncoderNumberCategories: Raised if the number of
		categories is too low.
		
		@raise InvalidCategoryEncoderNumberBits: Raised if the number of bits
		is less than the number of categories.
		"""
		
		# Format parameters appropriately
		self.num_categories = int(num_categories)
		if num_bits is not None:
			self.num_bits = int(num_bits)
			self.active_bits = self.num_bits / self.num_categories
			if active_bits is not None:
				if self.num_categories * int(active_bits) <= self.num_bits:
					self.active_bits = int(active_bits)
		else:
			self.active_bits = int(active_bits)
			self.num_bits    = self.num_categories * self.active_bits
		
		# Check the configuration options
		if self.num_categories < 1:
			raise InvalidCategoryEncoderNumberCategories(self.num_categories)
		if self.num_bits < self.num_categories:
			raise InvalidCategoryEncoderNumberBits(self.num_bits,
				self.num_categories)
		
		# Create the appropriate scalar encoder to represent the categories
		self.encoder = Scalar(num_bins=self.num_categories, bin_overlap=0,
			min_val=0, max_val=self.num_categories - 1,
			active_bits=self.active_bits, num_bits=self.num_bits, wrap=False,
			**kargs)
		
		self._valid_categories = set(range(self.num_categories))
	
	def _encode(self, value):
		"""
		Encodes a single value.
		
		@param value: The value to encode. If this value is None, the 0 state
		is returned.
		
		@return: The encoded value.
		"""
		
		if value is not None and value not in self._valid_categories:
			raise InvalidCategoryEncoderCategory(value,
				sorted(self._valid_categories))
		
		return self.encoder._encode(value)
	
	def _decode(self, encoding):
		"""
		Decode the provided value.
		
		@param encoding: The encoded bitstream.
		
		@return: The category that the encoding belongs to.
		"""
		
		d = self.encoder._decode(encoding)[0]
		return d if d is None else int(d)
	
	def get_all_inputs(self):
		"""
		Get all possible inputs. Note that there can be more inputs that will
		be created here, but these inputs will guarantee that each bin will be
		represented. To effectively use this method it should be noted that
		each input maps directly to a bin integer.
		
		@return: Returns a tuple containing a numpy array of a set of inputs
		that will encompass all encodings and a numpy array of sequential
		integers representing the range of the number of bits.
		"""
		
		return self.encoder.get_all_inputs()