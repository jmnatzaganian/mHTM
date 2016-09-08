# scalar.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 06/05/14
#	
# Description    : Implementation of a scalar encoder
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Implementation of a scalar encoder. This encoder quantizes scalars and then
performs a multi-hot encoding of each scalar.

G{packagetree nano_htm}
"""

__docformat__ = 'epytext'

# Third party imports
import numpy as np

# Program imports
from encoder import Encoder
from errors import BaseException, wrap_error

###############################################################################
########## Exception Handling
###############################################################################

class InvalidScalarEncoderLimits(BaseException):
	"""
	Exception if minimum value is greater than or equal to the maximum value.
	"""
	
	def __init__(self, min_value, max_value):
		"""
		Initialize this class.
		
		@param min_value: The minimum scalar encoder value.
		
		@param max_value: The maximum scalar encoder value.
		"""
		
		self.msg = wrap_error('The minimum value, {0}, is greater than or '
			'equal to the maximum value, {1}.'.format(min_value, max_value))

class InvalidScalarEncoderBinOverlap(BaseException):
	"""
	Exception if the bin overlap is >= 1.
	"""
	
	def __init__(self, bin_overlap):
		"""
		Initialize this class.
		
		@param bin_overlap: The percentage overlap of the bins in the scalar
		encoder.
		"""
		
		self.msg = wrap_error('The specified bin overlap, {0}, is greater '
			'than or equal to 1.'.format(bin_overlap))

class InvalidScalarEncoderBins(BaseException):
	"""
	Exception if the number of bins < 1.
	"""
	
	def __init__(self, num_bins):
		"""
		Initialize this class.
		
		@param num_bins: The number of bins to use.
		"""
		
		self.msg = wrap_error('The specified number of bins, {0}, is less '
			'than 1.'.format(num_bins))

class InvalidScalarEncoderBits(BaseException):
	"""
	Exception if the number of active bits < 1.
	"""
	
	def __init__(self, active_bits):
		"""
		Initialize this class.
		
		@param active_bits: The number of active bits.
		"""
		
		self.msg = wrap_error('The specified number of active bits, {0}, is '
			'less than 1.'.format(active_bits))

class InconsistentScalarEncoderState(BaseException):
	"""
	Exception if the encoded bins are not unique
	"""
	
	def __init__(self):
		"""
		Initialize this class.
		"""
		
		self.msg = wrap_error('The state of the scalar encoder is incosistent.'
			' This occurs when the specified parameters are poor. Try '
			'increasing the available encoding resources and trying again.'
			' If you continue to see this error, try decreasing the amount of '
			'bin_overlap.')

class InvalidScalarEncoderNumberBits(BaseException):
	"""
	Exception if the number of bits is too small.
	"""
	
	def __init__(self, nbits):
		"""
		Initialize this class.
		
		@param nbits: The number of bits to use.
		"""
		
		self.msg = wrap_error('The requested number of bits, {0}, is too'
			' low. Increase the number of bits and try again.'.format(nbits))

class InvalidScalarEncoderDecodeOption(BaseException):
	"""
	Exception if the desired decoding option is unsupported.
	"""
	
	def __init__(self, decode, valid_decodes):
		"""
		Initialize this class.
		
		@param decode: The desired decoding type.
		
		@param valid_decodes: The supported decoding types.
		"""
		
		self.msg = wrap_error('The requested decoding type, {0}, is invalid.'
			' Please choose a decoding type from the following: '.format(
			decode, ', '.join(sorted(valid_decodes))))

###############################################################################
########## Class Implementations
###############################################################################

class Scalar(Encoder):
	"""
	Scalar encoder. This encoder quantizes scalars and then performs a
	multi-hot encoding of each scalar.
	"""
	
	def __init__(self, num_bins, min_val, max_val, active_bits=None,
		num_bits=None, bin_overlap=0., wrap=False, decode_type="all", **kargs):
		"""
		Initializes this encoder.
		
		@param num_bins: The number of bins to use.
		
		@param min_val: The minimum value of the input.
		
		@param max_val: The maximum value of the input (must be > min_val).
		
		@param active_bits: The number of active bits in the number (bits that
		are '1'). If None, num_bits must be provided.
		
		@param num_bits: The number of bits to use. If None, this will be sized
		to fit. If not None, the active_bits will be scaled as needed.
		WARNING - If the number of bits is set too low, the bin_overlap
		parameter may be ignored. Similarly, this argument takes precedence
		over active_bits.
		
		@param bin_overlap: The maximum percentage that bins should overlap.
		
		@param wrap: Boolean to determine if the input should wrap around.
		
		@param decode_type: The type of decoding to return. The decoding refers
		to where the value lies in the bin. This may be as follows:
		1. "all" - returns a tuple of the following elements (median, min, max)
		2. "med" - returns the median
		3. "min" - returns the min
		4. "max" - returns the max
		
		@param kargs: Any keyword arguments to be passed to the base class.
		
		@raise InvalidScalarEncoderLimits: Raised if specified minimum limit is
		greater than the specified maximum limit.
		
		@raise InvalidScalarEncoderBinOverlap: Raised if the specified bin
		overlap is greater than or equal to one.
		
		@raise InvalidScalarEncoderBins: Raised if the total number of bins is
		less than 1.
		
		@raise InvalidScalarEncoderBits: Raised if the number of active bits is
		less than 1.
		
		@raise InvalidScalarEncoderNumberBits: Raised if the number of bits is
		too low.
		
		@raise InconsistentScalarEncoderState: Raised if the encoded bins are
		not all unique.
		
		@raise InvalidScalarEncoderDecodeOption: Raised if the desired decoding
		options is invalid.
		"""
		
		def simple_initialize():
			"""
			Initialize the scalar encoder using the simplified settings (where
			num_bits is not supplied).
			"""
			
			if self.num_bins == 1:
				self.step_size   = 1
				self.num_bits    = int(self.active_bits)
				self.bin_spacing = 1
			else:
				# The step size between bins
				self.step_size = self.active_bits - int(self.bin_overlap *
					self.active_bits)
				
				# The number of bits the scalar representation should have
				self.num_bits = self.num_bins * self.step_size
				if not self.wrap:
					self.num_bits += self.active_bits - self.step_size
				
				# The "distance" between bins
				self.bin_spacing = (self.max_val - self.min_val)              \
					/ (self.num_bins - 1.0)
		
		def initialize_params(num_bits=num_bits, active_bits=active_bits):
			"""
			Initialize the scalar encoder. This function handle all possible
			initialization configurations and ensures that the encoder will be
			created properly.
			
			@param num_bits: The desired number of bits.
			
			@param active_bit: The desired number of active bits.
			
			@raise InvalidScalarEncoderNumberBits: Raised if the number of bits
			becomes too small.
			"""
			
			self.pad = 0
			if num_bits is not None:
				num_bits = int(num_bits)
				if self.num_bins == 1:
					self.active_bits = num_bits / self.num_bins
					self.step_size   = 1
					self.bin_spacing = 1
					if active_bits is not None:
						active_bits = int(active_bits)
						if self.active_bits <= num_bits:
							self.active_bits = active_bits
							self.pad = num_bits - self.active_bits
				else:
					if active_bits is None:
						# Initialize active_bits to be as high as possible
						if not wrap:
							self.active_bits = int(num_bits / (self.num_bins *
								(1 - self.bin_overlap))) + 1
						else:
							self.active_bits = int(num_bits / (num_bins * ( 1 -
								bin_overlap) - bin_overlap)) + 1
					else:
						self.active_bits = int(active_bits)
					simple_initialize()
					while self.num_bits > num_bits:
						self.active_bits -= 1
						simple_initialize()
						if self.active_bits == 0:
							raise InvalidScalarEncoderNumberBits(num_bits)
					self.pad = num_bits - self.num_bits
					self.num_bits = num_bits
			else:
				if active_bits is None:
					raise InvalidScalarEncoderBits(active_bits)
				self.active_bits = int(active_bits)
				simple_initialize()
		
		# Store the input params
		self.num_bins    = int(num_bins)
		self.bin_overlap = float(bin_overlap)
		self.min_val     = float(min_val)
		self.max_val     = float(max_val)
		self.wrap        = bool(wrap)
		self.decode_type = decode_type.lower()
		
		initialize_params()
				
		# Check the configuration options
		if self.min_val >= self.max_val:
			raise InvalidScalarEncoderLimits(self.min_val, self.max_val)
		if self.bin_overlap < 0:
			self.bin_overlap = 0
		elif self.bin_overlap >= 1:
			raise InvalidScalarEncoderBinOverlap(self.bin_overlap)
		if self.num_bins < 1:
			raise InvalidScalarEncoderBins(self.num_bins)
		if self.active_bits < 1:
			raise InvalidScalarEncoderBits(self.active_bits)
		valid_decodes = set(['all', 'med', 'min', 'max'])
		if self.decode_type not in valid_decodes:
			raise InvalidScalarEncoderDecodeOption(self.decode_type,
				valid_decodes)
		
		# Check bin uniqueness
		bins = self.get_all_bins()
		l0, l1 = len(bins), len(set(tuple(x) for x in bins))
		if l0 != l1:
			raise InconsistentScalarEncoderState
		
		# Set decode_type slice
		if self.decode_type == 'all':
			self.slice = (0, 3)
		elif self.decode_type == 'med':
			self.slice = (0, 1)
		elif self.decode_type == 'min':
			self.slice = (1, 2)
		else:
			self.slice = (2, 3)
		
		super(Scalar, self).__init__(self.num_bits, **kargs)
	
	def _get_bin(self, value):
		"""
		Determines which bin the value lies in.
		
		@param value: The value to use.
		
		@return: The bin index or None if the value was None.
		"""
		
		if value is None:
			return None
		
		# Calculate ideal bin
		bin = (value - self.min_val) / self.bin_spacing
		
		# Move bin to closest neighboring bin
		if bin % 1 > 0.5:
			new_bin = int(bin) + 1
		else:
			new_bin = int(bin)
		
		# Return bounded bin
		if new_bin > self.num_bins - 1:
			return self.num_bins - 1
		elif new_bin < 0:
			return 0
		else:
			return new_bin
	
	def _encode_bin(self, bin):
		"""
		Get the encoding for the specified bin.
		
		@param bin: The bin being used.
		
		@return: The encoded representation.
		"""
		
		# Bitwise representation
		encoding = np.zeros(self.num_bits, dtype='bool')
		max_bits = self.num_bits - self.pad
		
		if bin is not None:
			# The starting index
			s_ix = bin * self.step_size
			# Build the bitwise representation
			if s_ix + self.active_bits <= max_bits:
				# No wrap, so set regular bits
				encoding[s_ix:s_ix + self.active_bits] = 1
			else:
				# Wrapping occurred
				num_bits                = self.active_bits - (max_bits - s_ix)
				encoding[s_ix:max_bits] = 1
				encoding[:num_bits]     = 1
		
		return encoding
	
	def _encode(self, value):
		"""
		Encodes a single value.
		
		@param value: The value to encode. If this value is None, the 0 state
		is returned.
		
		@return: The encoded value.
		"""
		
		return self._encode_bin(self._get_bin(value))
	
	def _decode(self, encoding):
		"""
		Decode the provided value. It is possible that there is no clear choice
		as to what the decoded value should be. Determining the correct winner
		goes as follows:
			1. Find the bin(s) with the most number of active bits
			2. Compute the average of the median, min, and max of each of those
			bins.
		
		If multiple bins are found, there is no way of knowing which one it
		should have been; therefore, the average of all possibilities is used.
		
		@param encoding: The encoded bitstream.
		
		@return: A tuple containing the median value for the encoding, the
		minimum value, and the maximum value, respectively. It is possible that
		the actual encoded value was less than or greater than the reported min
		and max, respectively; thus, the returned values should be taken as
		what can be reliably encoded / decoded. This function may return a
		tuple of None. This occurs when there are no active bits in the
		encoding.
		"""
				
		# Check if the value exists
		if np.sum(encoding) == 0:
			ret = [None, None, None][self.slice[0]:self.slice[1]]
		else:
			# Get a list of (number of active bits, bin) for all bins sorted by
			# the largest value
			bins = sorted([(np.sum(np.bitwise_and(bin, encoding)), i) for
				i, bin in enumerate(self.get_all_bins())], reverse=True)
			
			# Create a new list only containing the best matching bins
			best_bins = np.array([bin for nbits, bin in bins
				if nbits == bins[0][0]])
			
			# Get the average values
			medians = best_bins * self.bin_spacing + self.min_val
			mins    = medians - self.bin_spacing / 2
			maxs    = medians + self.bin_spacing / 2
			
			# Prep output
			ret = [np.mean(medians), np.mean(mins), np.mean(maxs)][
				self.slice[0]:self.slice[1]]
		
		# Return single value or tuple, based off what was desired
		if len(ret) == 1:
			return ret[0]
		else:
			return tuple(ret)
	
	def encoding_to_str(self, encoding):
		"""
		Get a string representation of the encoding.
		
		@param encoding: The encoding to print.
		
		@return: A string representing the encoding.
		"""
		
		fmt = lambda i: '-' if i % self.step_size == 0 and i != 0 else ''
		return ''.join('{0}1'.format(fmt(i)) if r == 1 else
			'{0}.'.format(fmt(i)) for i, r in enumerate(encoding))
	
	def get_all_bins(self):
		"""
		Get all possible encodings.
		
		@return: A list of all possible encodings.
		"""
		
		bins = []
		for bin in xrange(self.num_bins):
			bins.append(self._encode_bin(bin))
		return bins
	
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
		
		return (np.array([self._decode(b) for b in self.get_all_bins()]),
			np.arange(self.num_bins))