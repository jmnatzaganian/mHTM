# mnist.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 10/13/15
#	
# Description    : Testing SP with MNIST.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Testing SP with MNIST.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import cPickle, time, os
from itertools import izip

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from scipy.misc import imread
import matplotlib.gridspec as gridspec
from sklearn.feature_extraction.image import extract_patches_2d

# Program imports
from mHTM.region import SPRegion, load
from mHTM.datasets.loader import load_mnist

def plot_weights(input, weights, nrows, ncols, shape, out_path=None,
	show=True):
	"""
	Make a video showing what happens with the input.
	"""
	
	# Construct the basic plot
	fig = plt.figure(figsize=(16, 8))
	gs = gridspec.GridSpec(nrows, ncols + 2, wspace=0, hspace=0.5)
	
	ax = plt.subplot(gs[nrows/2, 0])
	ax.imshow(input.reshape(shape), cmap=plt.get_cmap('gray'), vmin=0,
		vmax=1, interpolation='none')
	ax.axis('off')
	
	ax = plt.subplot(gs[nrows/2, 1])
	ax.imshow(imread('arrow2.png'), cmap=plt.get_cmap('gray'))
	ax.axis('off')
	
	# Add all of the figures to the grid
	for i, weight_set in enumerate(weights):
		row = i / ncols
		col = i + 2 - row * ncols
		ax = plt.subplot(gs[row, col])
		ax.imshow(weight_set.reshape(shape), cmap=plt.get_cmap('gray'), vmin=0,
			vmax=1, interpolation='none')
		ax.axis('off')
	fig.patch.set_visible(False)
	
	# Save the plot
	# fig.set_size_inches(19.20, 10.80)
	if out_path is not None:
		# plt.savefig(out_path, format=out_path.split('.')[-1], dpi = 100)
		plt.savefig(out_path, format=out_path.split('.')[-1])
	
	# Show the plot and close it after the user is done
	if show: plt.show()
	plt.close()

def fit_grid():
	"""
	Use a grid technique with many SPs.
	"""
	
	p = 'results\\mnist_filter'
	# try:
		# os.makedirs(p)
	# except OSError:
		# pass
	np.random.seed(123456789)
	# kargs = {
		# 'ninputs': 9,
		# 'ncolumns': 100,
		# 'nsynapses': 5,
		# 'random_permanence': True,
		# 'pinc':0.03, 'pdec':0.05,
		# 'seg_th': 3,
		# 'nactive': 10,
		# 'duty_cycle': 100,
		# 'max_boost': 10,
		# 'global_inhibition': True,
		# 'trim': 1e-4
	# }
	kargs2 = {
		'ninputs': 100 * (26 ** 2),
		'ncolumns': 2048,
		'nsynapses': 1000,
		'random_permanence': True,
		'pinc':0.03, 'pdec':0.05,
		'seg_th': 5,
		'nactive': 20,
		'duty_cycle': 100,
		'max_boost': 10,
		'global_inhibition': True,
		'trim': 1e-4
	}
	
	# Get the data
	(tr_x, tr_y), (te_x, te_y) = get_data()
	nwindows = 26 ** 2
	
	# # Make the SPs
	# sps = [SPRegion(**kargs) for _ in xrange(nwindows)]
	
	# # Train the SPs
	# nepochs = 10
	# t = time.time()
	# for i in xrange(nepochs):
		# print i
		# for j, x in enumerate(tr_x):
			# print '\t{0}'.format(j)
			# nx = extract_patches_2d(x.reshape(28, 28), (3, 3)).reshape(
				# nwindows, 9)
			# for xi, sp in izip(nx, sps):
				# sp.step(xi)
	# t1 = time.time() - t
	# print t1
	
	# # Save this batch of SPs
	# for i, sp in enumerate(sps):
		# sp.learn = False
		# sp.save(os.path.join(p, 'sp0-{0}.pkl'.format(i)))
	
	# Make the top level SP
	sp2 = SPRegion(**kargs2)
	
	# Get the SPs
	sps = [load(os.path.join(p, sp)) for sp in os.listdir(p) if sp[2] == '0']
	
	# Train the top SP
	nepochs = 10
	t = time.time()
	for i in xrange(nepochs):
		print i
		for j, x in enumerate(tr_x):
			print '\t{0}'.format(j)
			nx = extract_patches_2d(x.reshape(28, 28), (3, 3)).reshape(
				nwindows, 9)
			output = np.array(np.zeros(100 * nwindows), dtype='bool')
			for k, (xi, sp) in enumerate(izip(nx, sps)):
				sp.step(xi)
				output[k*100:(k*100)+100] = sp.y[:, 0]
			sp2.step(output)
	t2 = time.time() - t
	print t2
	
	# Save the top SP
	sp2.learn = False
	sp2.save(os.path.join(p, 'sp1-0.pkl'))

def score_grid():
	"""
	Classify with the gridded SP.
	"""
	
	p = 'results\\mnist_filter'
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	
	# Get the SPs
	sps = [load(os.path.join(p, sp)) for sp in os.listdir(p) if sp[2] == '0']
	sp2 = load(os.path.join(p, 'sp1-0.pkl'))
	
	nwindows = 26 ** 2
	nfeat = 100 * nwindows
	
	# w = [sp2.p[sp2.syn_map == j] for j in xrange(nfeat)]
	# ms = max(wi.shape[0] for wi in w)
	# with open(os.path.join(p, 'data.pkl'), 'wb') as f:
		# cPickle.dump((w, ms), f, cPickle.HIGHEST_PROTOCOL)
	with open(os.path.join(p, 'data.pkl'), 'rb') as f:
		w, ms = cPickle.load(f)
	
	# Get training data
	tr_x2 = np.zeros((tr_x.shape[0], nfeat))
	for i, x in enumerate(tr_x):
		nx = extract_patches_2d(x.reshape(28, 28), (3, 3)).reshape(
			nwindows, 9)
		x = np.array(np.zeros(nfeat), dtype='bool')
		for j, (xi, sp) in enumerate(izip(nx, sps)):
			sp.step(xi)
			x[j*100:(j*100)+100] = sp.y[:, 0]
		
		y = sp2.p * x[sp2.syn_map]
		w = np.zeros((nfeat, ms))
		for j in xrange(nfeat):
			a = y[sp2.syn_map == j]
			w[j][:a.shape[0]] = a
		tr_x2[i] = np.mean(w, 1)
	
	# Get testing data
	te_x2 = np.zeros((te_x.shape[0], nfeat))
	for i, x in enumerate(te_x):
		nx = extract_patches_2d(x.reshape(28, 28), (3, 3)).reshape(
			nwindows, 9)
		x = np.array(np.zeros(nfeat), dtype='bool')
		for j, (xi, sp) in enumerate(izip(nx, sps)):
			sp.step(xi)
			x[j*100:(j*100)+100] = sp.y[:, 0]
		
		y = sp2.p * x[sp2.syn_map]
		w = np.zeros((nfeat, ms))
		for j in xrange(nfeat):
			a = y[sp2.syn_map == j]
			w[j][:a.shape[0]] = a
		te_x2[i] = np.mean(w, 1)
	
	# Classify
	clf = LinearSVC(random_state=123456789)
	clf.fit(tr_x2, tr_y)
	print 'SVM Accuracy : {0:2.2f} %'.format(clf.score(te_x2, te_y) * 100)

if __name__ == '__main__':
	pass