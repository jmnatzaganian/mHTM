# base.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 10/25/14
#	
# Description    : The base class description for an encoder
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Skeleton of an encoder.

G{packagetree nano_htm}
"""

__docformat__ = 'epytext'

# Native imports
from abc import ABCMeta, abstractmethod

# Third party imports
import numpy as np

# Program imports
from util import flatten, create_unique_name

###############################################################################
########## Class Template
###############################################################################

class Encoder(object):
	"""
	Base class for an encoder.
	"""
	__metaclass__ = ABCMeta
	
	def __init__(self, num_bits, name=None):
		"""
		Initializes the encoder.
		
		@param num_bits: The number of bits that the encoded value will be.
		
		@param name: The name of the encoder.
		"""
		
		self.num_bits = num_bits
		if name is None:
			self.name = create_unique_name()
		else:
			self.name = name
	
	@abstractmethod
	def _encode(self, value):
		"""
		Encodes a single value.
		
		@param value: The value to encode.
		
		@return: The encoded value.
		"""
	
	@abstractmethod
	def _decode(self, value):
		"""
		Decodes a single value.
		
		@param value: The value to decode.
		
		@return: The decoded value.
		"""
	
	def encode(self, data=None):
		"""
		Use a generator to yield each encoded bit. This supports being able to
		encode a list of values, where each value will be sequentially encoded.
		This function can also encode single values.
		
		@param data: The data to encode. If it isn't provided the encoder's
		data is used.
		"""
		
		if data is None:
			data = self.raw_data
		
		# Make a list to account for single inputs
		for x in flatten([data]):
			for bit in flatten([self._encode(x)]):
				yield bit
	
	def decode(self, data=None):
		"""
		Decode the data.
		
		Note - It is possible that an encoder will not be able to decode the
		input. In this event, the encoded data will be returned.
		
		@param data: The data to decode. If it isn't provided the encoder's
		data is used.
		
		@return: The decoded data.
		"""
		
		if data is None:
			data = self.raw_data
		
		return self._decode(np.array(data))
	
	def encoding_to_str(self, encoding):
		"""
		Get a string representation of the encoding.
		
		@param encoding: The encoding to print.
		
		@return: A string representing the encoding.
		"""
		
		return ''.join('{0}'.format(bit) for bit in encoding)
	
	def bind_data(self, raw_data):
		"""
		Binds the raw data to the encoder. The raw_data must be the current
		value to encode. If this a list, the output will be a concatenation of
		all encoded bits. If it is a single value, the output will be the
		encoding of just that value.
		
		@param raw_data: The data to bind.
		"""
		
		self.raw_data = raw_data
	
	def free_data(self):
		"""
		Deletes the data stored in the encoder.
		"""
		
		del self.raw_data