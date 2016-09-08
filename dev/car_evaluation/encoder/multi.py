# multi.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 06/09/15
#	
# Description    : Implementation of a multi encoder
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Implementation of a multi encoder. This encoder combines mulitple encoders into
a single encoder.

G{packagetree nano_htm}
"""

__docformat__ = 'epytext'

# Third party imports
import numpy as np

# Program imports
from encoder import Encoder

class Multi(Encoder):
	"""
	Multi encoder. This encoder combines multiple encoders together. The final
	encoding is a concatenation of outputted streams.
	"""
	
	def __init__(self, *encoders, **kargs):
		"""
		Initializes this encoder.
		
		@param encoders: One or more encoders.
		
		@param kargs: Any keyword arguments to be passed to the base class.
		"""
		
		self.encoders = encoders
		super(Multi, self).__init__(sum(e.num_bits for e in self.encoders),
			**kargs)
		
		# Mapping of starting and ending indexes for each encoder
		self.indexes = {}
		s = e = 0
		for i, encoder in enumerate(self.encoders):
			e = s + encoder.num_bits
			self.indexes[i] = (s, e)
			s = e
	
	def _encode(self, values):
		"""
		Encodes each value with the specified encoder.
		
		@param values: One or more tuples containing (value, index) pairs,
		where the index refers to which encoder to use. For any missing
		encoders 0's will be used.
		
		@return: The encoded representation.
		"""
		
		encoding = np.zeros(self.num_bits, dtype='bool')
		if values is not None:
			for value, ix in values:
				encoding[self.indexes[ix][0]:self.indexes[ix][1]] =           \
					self.encoders[ix]._encode(value)
		return encoding
	
	def _decode(self, encoding):
		"""
		Decodes the encoding.
		
		@param encoding: The encoded value.
		
		@return: A list containing the decoding for each portion of the encoded
		sequence.
		"""
		
		decodings = []
		for ix, encoder in enumerate(self.encoders):
			decodings.append(encoder._decode(
				encoding[self.indexes[ix][0]:self.indexes[ix][1]]))
		return decodings
		
	def encode(self, data=None):
		"""
		Use a generator to yield each encoded bit. This supports being able to
		encode a list of values, where each value will be sequentially encoded.
		This function can also encode single values.
		
		@param data: The data to encode. If it isn't provided the encoder's
		data is used.
		"""
		
		if data is None:
			data = [(encoder.raw_data, ix) for ix, encoder in enumerate(
				self.encoders)]
		
		for bit in self._encode(data):
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
			data = [encoder.raw_data for encoder in self.encoders]
		
		return self._decode(np.array(data))
		
	def encoding_to_str(self, encoding):
		"""
		Get a string representation of the encoding.
		
		@param encoding: The encoding to print.
		
		@return: A string representing the encoding.
		"""
		
		strs = []
		for ix, encoder in enumerate(self.encoders):
			strs.append(encoder.encoding_to_str(
				encoding[self.indexes[ix][0]:self.indexes[ix][1]]))
		return ''.join(strs)
	
	def bind_data(self, raw_data):
		"""
		Binds the raw data to the encoder.
		
		@param raw_data: One or more tuples containing (value, index) pairs,
		where the index refers to which encoder to use. For any missing
		encoders 0's will be used.
		"""
		
		# Initialize data to be 0
		for encoder in self.encoders:
			encoder.bind_data(None)
		
		# Bind all data
		if raw_data is not None:
			for data, ix in raw_data:
				self.encoders[ix].bind_data(data)
	
	def free_data(self):
		"""
		Deletes the data stored in the encoder.
		"""
		
		for encoder in self.encoders:
			encoder.free_data()