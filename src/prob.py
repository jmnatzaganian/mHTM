# prob.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 11/12/15
#	
# Description    : Module for computing various probabilities
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Module for computing various probabilities

G{packagetree mHTM}
"""

# Third-Party imports
import numpy as np

# Program imports
from mHTM.region import SPRegion

###############################################################################
#### Accessory functions
###############################################################################

def comb(n, k):
	"""
	Compute the binomial coefficients.
	
	@param n: The total number of elements.
	
	@param k: The number of selections.
	
	@return: The number of times k items may be chosen from a set of size n.
	"""
	
	p = 1.
	for i in xrange(1, k + 1):
		p *= (n + 1. - i) / i
	return p

###############################################################################
#### Probabilities
###############################################################################

def p_a1(npsyns, ninputs):
	"""
	Probability of selecting one input given ninputs and npsyns attempts. This
	uses the intersection operator.
	
	@param npsyns: The number of proximal synapses.
	
	@param ninputs: The number of inputs.
	
	@return: The computed probability.
	"""
	
	return (npsyns + 1.) / ninputs

def p_a2(npsyns, ninputs):
	"""
	Probability of selecting one input given ninputs and npsyns attempts. This
	uses a binomial distribution.
	
	@param npsyns: The number of proximal synapses.
	
	@param ninputs: The number of inputs.
	
	@return: The computed probability.
	"""
	
	p = 1. / ninputs
	return npsyns * p * ((1 - p) ** (npsyns - 1))

def p_b(ncols, npsyns, ninputs, k=1):
	"""
	Probability of one input connecting to at least k columns.
	
	@param ncols: The number of columns.
	
	@param npsyns: The number of proximal synapses.
	
	@param ninputs: The number of inputs.
	
	@param k: The minimum number of columns that an input must connect to.
	
	@return: The computed probability.
	"""
	
	# Probability of an input connecting to a column
	p = p_a1(npsyns, ninputs)
	
	# Sum of probabilities of an input connecting to k columns
	p2 = 0.
	for i in xrange(k):
		p2 += comb(ncols, i) * (p ** i) * ((1 - p) ** (ncols - i))
	return p2

def p_c(ncols, npsyns, ninputs):
	"""
	Probability of an input never being selected.
	
	@param ncols: The number of columns.
	
	@param npsyns: The number of proximal synapses.
	
	@param ninputs: The number of inputs.
		
	@return: The computed probability.
	"""
	
	p0 = p_a1(npsyns, ninputs)
	p = (1 - p0) ** ncols
	return p

###############################################################################
#### Expected Values
###############################################################################

def e_b(ncols, npsyns, ninputs):
	"""
	Expected number of columns that each input will be connected to.
	
	@param ncols: The number of columns.
	
	@param npsyns: The number of proximal synapses.
	
	@param ninputs: The number of inputs.
	
	@return: The computed mean.
	"""
	
	return ncols * p_a1(npsyns, ninputs)

def e_c(npsyns, p_active_input):
	"""
	Expected number of active synapses on a column.
	
	@param npsyns: The number of proximal synapses.
	
	@param p_active_input: The probability that a given input bit will
	be active.
	
	@return: The computed mean.
	"""
	
	return npsyns * p_active_input

def e_d(npsyns, p_active_input, syn_th):
	"""
	Expected number of active connected synapses on a column.
	
	@param npsyns: The number of proximal synapses.
	
	@param p_active_input: The probability that a given input bit will
	be active.
	
	@param syn_th: The threshold at which synapses are connected.
	
	@return: The computed mean.
	"""
	
	return e_c(npsyns, p_active_input) * syn_th

def e_e(npsyns, p_active_input, ncols, seg_th):
	"""
	Expected number of columns with active inputs >= seg_th.
	
	@param npsyns: The number of proximal synapses.
	
	@param p_active_input: The probability that a given input bit will
	be active.
	
	@param ncols: The number of columns.
	
	@param seg_th: The threshold for a segment to become active.
	
	@return: The computed mean.
	"""
	
	p = p_active_input
	p2 = 0.
	for i in xrange(seg_th):
		p2 += comb(npsyns, i) * (p ** i) * ((1 - p) ** (npsyns - i))
	return ncols * (1 - p2)

def e_f(npsyns, p_active_input, ncols, seg_th, syn_th):
	"""
	Expected number of columns with active connected inputs >= seg_th.
	
	@param npsyns: The number of proximal synapses.
	
	@param p_active_input: The probability that a given input bit will
	be active.
	
	@param ncols: The number of columns.
	
	@param seg_th: The threshold for a segment to become active.
	
	@param syn_th: The threshold at which synapses are connected.
	
	@return: The computed mean.
	"""
	
	p = p_active_input * syn_th
	
	p2 = 0.
	for i in xrange(seg_th):
		p2 += comb(npsyns, i) * (p ** i) * ((1 - p) ** (npsyns - i))
	return ncols * (1 - p2)

def main(ncols, npsyns, ninputs, density, seg_th, syn_th, ntrials=100,
	seed=123456789):
	"""
	Compare the theoretical to the experimental results.
	
	@param ncols: The number of columns.
	
	@param npsyns: The number of proximal synapses.
	
	@param ninputs: The number of inputs.
	
	@param density: The percentage of active bits in the input.
	
	@param seg_th: The threshold for a segment to become active.
	
	@param syn_th: The threshold at which synapses are connected.
	
	@param ntrials: The number of trials to perform for the experimental
	results.
	
	@param seed: Seed for the random number generator for 
	"""
	
	print '**** THEORETICAL ****'
	print 'Probability that an input will be selected: {0:2.2f}%'.format(
		p_a1(npsyns, ninputs) * 100)
	p = p_c(ncols, npsyns, ninputs)
	print 'Probability of all inputs being selected: {0:2.2f}%'.format((1 - p)
		* 100)
	print 'Expected inputs not seen:', int(p * ninputs )
	print 'Expected number of columns connected to an input:', int(e_b(ncols,
		npsyns, ninputs))
	print 'Expected number of active synapses on a column: {0:2.2f}'.format(
		e_c(npsyns, density))
	print 'Expected number of active connected synapses on a column: ' \
		'{0:2.2f}'.format(e_d(npsyns, density, syn_th))
	print 'Expected number of columns with active inputs >= seg_th: {0:2.2f}' \
		.format(e_e(npsyns, density, ncols, seg_th))
	print 'Expected number of columns with active connected inputs >= ' \
		'seg_th: {0:2.2f}'.format(e_f(npsyns, density, ncols, seg_th,
		syn_th))
	
	# Prep the experimental
	print '\n**** Experimental ****'
	np.random.seed(seed)
	kargs = {
		'ninputs': ninputs,
		'ncolumns': ncols,
		'nsynapses': npsyns,
		'syn_th': syn_th,
		'seg_th': seg_th
	}
	
	#### Average number of active bits potentially connected to a column
	# Build input
	x = np.zeros(ninputs, dtype='bool')
	nactive = int(ninputs * (density))
	indexes = set(np.random.randint(0, ninputs, nactive))
	while len(indexes) != nactive:
		indexes.add(np.random.randint(0, ninputs - 1, 1)[0])
	x[list(indexes)] = True
	
	# Simulate
	y0 = y1 = y2 = y3 = y4 = y5 = 0.
	for _ in xrange(ntrials):
		sp = SPRegion(**kargs)
		a = x[sp.syn_map].sum(1)
		b = (x[sp.syn_map] * (sp.p >= syn_th)).sum(1)
		y0 += ninputs - len(set(sp.syn_map.ravel()))
		y1 += np.mean(np.array([np.sum(sp.syn_map == i) for i in
			xrange(ninputs)]))
		y2 += a.mean()
		y3 += b.mean()
		y4 += (a >= seg_th).sum()
		y5 += (b >= seg_th).sum()
	print 'Average number of missing inputs: {0:.2f}'.format(y0 / ntrials)
	print 'Average number of columns connected to an input: {0:.2f}'.format(
		y1 / ntrials)
	print 'Average number of active inputs per column: {0:.2f}'.format(y2
		/ ntrials)
	print 'Average number of active connected inputs per column: {0:.2f}' \
		.format(y3 / ntrials)
	print 'Number of columns with active inputs >= seg_th {0:.2f}'.format(
		y4 / ntrials)
	print 'Number of columns with active connected inputs >= seg_th {0:.2f}' \
		.format(y5 / ntrials)

if __name__ == '__main__':
	ncols, npsyns, ninputs = 100, 75, 748
	density, seg_th, syn_th = 0.13, 2, 0.5
	ntrials, seed = 100, 123456789
	main(ncols, npsyns, ninputs, density, seg_th, syn_th, ntrials, seed)