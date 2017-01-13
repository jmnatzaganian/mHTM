# plot.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 10/30/15
#	
# Description    : Plotting module.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Plotting module.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import os
from itertools import cycle, izip

# Third-Party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Set the defaults
params = {
		'backend'              : 'ps',
		# 'text.usetex'        : True,
		# 'text.latex.preamble'  : ['\usepackage{gensymb}'],
		'axes.labelsize'       : 40,
		'axes.labelweight'     : 'bold',
		'axes.linewidth'       : 4,
		'axes.titlesize'       : 40,
		'axes.titleweight'     : 'bold',
		'font.size'            : 40,
		'legend.fontsize'      : 40,
		'font.weight'          : 'bold',
		'xtick.labelsize'      : 40,
		'ytick.labelsize'      : 40,
		'font.family'          : 'serif',
		'lines.linewidth'      : 4,
		'lines.markersize'     : 16,
		'xtick.major.size'     : 12,
		'ytick.major.size'     : 12,
		'xtick.major.width'    : 4,
		'ytick.major.width'    : 4,
		'figure.dpi'           : 100,
		'figure.figsize'       : (20, 12),
		'legend.fancybox'      : True
	}
rcParams.update(params)

###############################################################################
#### Helper Functions
###############################################################################

def compute_axis_bounds(x, multiplier=1):
	"""
	Compute the bounds for the axis. It is assumed that the data starts at 0,
	unless the data goes below zero. The bounds are based off the standard
	deviation.
	
	@param x: The data to apply to the axis. If multiple values are supplied,
	the largest bounds are used.
	
	@param multiplier: The multiplier to scale by. Recommended is 6 for Y axis
	and 1/6 for X axis.
	"""
	
	def compute_bounds(v):
		min_v = np.min(v)
		max_v = np.max(v)
		std = multiplier / 5. if min_v == max_v else np.std(v)
		lb = 0 if min_v > 0 else min_v - std
		ub = max_v + std
		return lb, ub
	
	bounds = np.array([compute_bounds(y) for y in x])
	return np.min(bounds.T[0]), np.max(bounds.T[1])

def compute_err(x, x_med=None, axis=1):
	"""
	Compute the relative errors based off IQR.
	
	@param x: A 2D numpy array containing the data across multiple iterations.
	Each row represents a unique instance of the data.
	
	@param x_med: A 1D numpy array representing the median values.
	
	@param axis: The axis around which to compute the data.
	
	@return: A 2D numpy array containing the -y and +y errors.
	"""
	
	if x_med is None:
		x_med = np.median(x, axis)
	
	return np.array([x_med - np.percentile(x, 25, axis),
		np.percentile(x, 75, axis) - x_med])

###############################################################################
#### 2D Plots
###############################################################################

def plot_epoch(y_series, series_names=None, y_errs=None, y_label=None,
	title=None, semilog=False, out_path=None, show=True, start_epoch=1):
	"""
	Basic plotter function for plotting various types of data against
	training epochs. Each item in the series should correspond to a single
	data point for that epoch.
	
	@param y_series: A tuple containing all of the desired series to plot.
	
	@param series_names: A tuple containing the names of the series.
	
	@param y_errs: The error in the y values. There should be one per series
	per datapoint. It is assumed this is the standard deviation, but any error
	will work.
	
	@param y_label: The label to use for the y-axis.
	
	@param title: The name of the plot.
	
	@param semilog: If True the y-axis will be plotted using a log(10) scale.
	
	@param out_path: The full path to where the image should be saved. The file
	extension of this path will be used as the format type. If this value is
	None then the plot will not be saved, but displayed only.
	
	@param show: If True the plot will be show upon creation.
	
	@param start_epoch: The epoch to start plotting at.
	"""
	
	x_series = [np.arange(start_epoch, y.shape[0] + start_epoch) for y in
		y_series]
	
	plot_error(x_series, y_series, series_names, y_errs, 'Epoch', y_label,
		title, semilog, out_path, show, ylim=(-5, 105))

def plot_error(x_series, y_series, series_names=None, y_errs=None,
	x_label=None, y_label=None, title=None, semilog=False, out_path=None,
	show=True, xlim=None, ylim=None, ax=None, legend=True):
	"""
	Basic plotter function for plotting data with error bars.
	
	@param x_series: A tuple containing all of the desired series to plot.
	
	@param y_series: A tuple containing all of the desired series to plot.
	
	@param series_names: A tuple containing the names of the series.
	
	@param y_errs: The error in the y values. There should be one per series
	per datapoint. It is assumed this is the standard deviation, but any error
	will work.
	
	@param y_label: The label to use for the x-axis.
	
	@param y_label: The label to use for the y-axis.
	
	@param title: The name of the plot.
	
	@param semilog: If True the y-axis will be plotted using a log(10) scale.
	
	@param out_path: The full path to where the image should be saved. The file
	extension of this path will be used as the format type. If this value is
	None then the plot will not be saved, but displayed only.
	
	@param show: If True the plot will be show upon creation.
	
	@param xlim: A tuple containing the bounds to use for the x-axis;
	otherwise, if None, it is automatically scaled. If False the default is
	used.
	
	@param ylim: A tuple containing the bounds to use for the y-axis;
	otherwise, if None, it is automatically scaled. If False the default is
	used.
	
	@param ax: The axis to use.
	
	@param legend: Boolean denoting if a legend should be made.
	"""
	
	# Construct the basic plot
	if ax is None:
		fig = plt.figure(figsize=(21, 10))
		ax = plt.subplot(111)
	else:
		fig = None
	if title is not None   : ax.set_title(title, y=1.03)
	if semilog             : ax.set_yscale('log')
	if x_label is not None : ax.set_xlabel(x_label)
	if y_label is not None : ax.set_ylabel(y_label)
	if xlim is not False:
		if xlim is not None:
			plt.xlim(*xlim)
		else:
			plt.xlim(*compute_axis_bounds(x_series, 1/6.))
	if ylim is not False:
		if ylim is not None:
			plt.ylim(*ylim)
		else:
			plt.ylim(*compute_axis_bounds(y_series, 6))
	colors  = cycle(['g', 'b', 'r', 'c', 'm', 'y', 'k'])
	markers = cycle(mlines.Line2D.filled_markers)
	
	# Add the data
	if y_errs is not None:
		for x, y, err in zip(x_series, y_series, y_errs):
			ax.errorbar(x, y, yerr=err, color=colors.next(), capsize=10,
				capthick=2,	marker=markers.next())
			ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
	else:
		for x, y in zip(x_series, y_series):
			ax.scatter(x, y, color=colors.next(), marker=markers.next(), s=120)
	
	# Create the legend
	if series_names is not None and legend:
		box = ax.get_position()
		ax.set_position([box.x0, box.y0 + box.height * 0.35, box.width,
			box.height * 0.65])
		leg = ax.legend(series_names, loc='upper center',
			bbox_to_anchor=(0, -1.25, 1, 1), ncol=4, mode='expand',
			borderaxespad=0.)
		leg.get_frame().set_linewidth(2)
	
	# Save the plot
	plt.subplots_adjust(bottom=0.3, hspace=0.3)
	if fig is not None:
		fig.set_size_inches(19.20, 10.80)
	if out_path is not None:
		plt.savefig(out_path, format=out_path.split('.')[-1], dpi = 100)
	
	# Show the plot and close it after the user is done
	if show: plt.show()
	# plt.close()
	
	if fig is not None:
		return fig, ax

def plot_line(x_series, y_series, series_names=None, x_label=None,
	y_label=None, title=None, semilog=False, out_path=None, show=True,
	xlim=None, ylim=None):
	"""
	Basic plotter function for plotting a line graph.
	
	@param x_series: A tuple containing all of the desired series to plot.
	
	@param y_series: A tuple containing all of the desired series to plot.
	
	@param series_names: A tuple containing the names of the series.
	
	@param y_label: The label to use for the x-axis.
	
	@param y_label: The label to use for the y-axis.
	
	@param title: The name of the plot.
	
	@param semilog: If True the y-axis will be plotted using a log(10) scale.
	
	@param out_path: The full path to where the image should be saved. The file
	extension of this path will be used as the format type. If this value is
	None then the plot will not be saved, but displayed only.
	
	@param show: If True the plot will be show upon creation.
	
	@param xlim: A tuple containing the bounds to use for the x-axis;
	otherwise, if None, it is automatically scaled. If False the default is
	used.
	
	@param ylim: A tuple containing the bounds to use for the y-axis;
	otherwise, if None, it is automatically scaled. If False the default is
	used.
	"""
	
	# Construct the basic plot
	fig, ax = plt.subplots()
	if title is not None   : plt.title(title, y=1.03)
	if semilog             : ax.set_yscale('log')
	if x_label is not None : ax.set_xlabel(x_label)
	if y_label is not None : ax.set_ylabel(y_label)
	if xlim is not False:
		if xlim is not None:
			plt.xlim(*xlim)
		else:
			plt.xlim(*compute_axis_bounds(x_series, 1/6.))
	if ylim is not False:
		if ylim is not None:
			plt.ylim(*ylim)
		else:
			plt.ylim(*compute_axis_bounds(y_series, 6))
	colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
	
	# Add the data
	for x, y in zip(x_series, y_series):
		ax.plot(x, y, color=colors.next(), linewidth=3)
	
	# Create the legend
	if series_names is not None:
		box = ax.get_position()
		ax.set_position([box.x0, box.y0 + box.height * 0.35, box.width,
			box.height * 0.65])
		leg = ax.legend(series_names, loc='upper center',
			bbox_to_anchor=(0, -1.22, 1, 1), ncol=4, mode='expand',
			borderaxespad=0.)
		leg.get_frame().set_linewidth(2)
	
	# Save the plot
	fig.set_size_inches(19.20, 10.80)
	if out_path is not None:
		plt.savefig(out_path, format=out_path.split('.')[-1], dpi = 100)
	
	# Show the plot and close it after the user is done
	if show: plt.show()
	plt.close()

###############################################################################
#### Image Plots
###############################################################################

def plot_image(x, shape, title=None, out_path=None, show=True):
	"""
	Plot a grayscale image. It is assumed that the image has been normalized
	(0, 1).
	
	@param x: A numpy array containing the image.
	
	@param shape: The shape of the data. This should be a tuple of x and y
	dimensions.
	
	@param title: The name of the plot.
	
	@param out_path: The full path, including the extension, of where to save
	the plot. If this paramter is None, the image will not be saved.
	
	@param show: If True the image will be shown.
	"""
	
	# Construct the basic plot
	fig = plt.figure()
	if title is not None: fig.suptitle(title, fontsize=16)
	
	# Add all of the figures to the grid
	plt.imshow(x.reshape(shape), cmap=plt.get_cmap('gray'), vmin=0, vmax=1,
		interpolation='none')
	plt.gca().axes.get_xaxis().set_visible(False)
	plt.gca().get_yaxis().set_visible(False)
	
	# Save the plot
	fig.set_size_inches(19.20, 10.80)
	if out_path is not None:
		plt.savefig(out_path, format=out_path.split('.')[-1], dpi = 100)
	
	# Show the plot and close it after the user is done
	if show: plt.show()
	plt.close()

def plot_grid_images(X, shape, nrows, ncols, title=None, labels=None,
	out_path=None, show=True):
	"""
	Construct a grid of grayscale images. It is assumed that the images have
	been normalized (0, 1).
	
	@param X: A numpy array containing a set of images.
	
	@param shape: The shape of the data. This should be a tuple of x and y
	dimensions.
	
	@param nrows: The number of rows the grid should have.
	
	@param ncols: The number of columns the grid should have.
	
	@param title: The name of the plot.
	
	@param labels: The labels for each image. If None, no labels will be made.
	
	@param out_path: The full path, including the extension, of where to save
	the plot. If this parameter is None, the image will not be saved.
	
	@param show: If True the image will be shown.
	"""
	
	# Construct the basic plot
	fig = plt.figure()
	if title is not None: fig.suptitle(title, fontsize=16)
	if labels is None: labels = [None] * X.shape[0]
	
	# Add all of the figures to the grid
	for i, (x, y) in enumerate(izip(X, labels)):
		ax = plt.subplot(nrows, ncols, i + 1)
		ax.set_title(y)
		ax.imshow(x.reshape(shape), cmap=plt.get_cmap('gray'), vmin=0,
			vmax=1, interpolation='none')
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
	
	# Save the plot
	fig.set_size_inches(19.20, 10.80)
	if out_path is not None:
		plt.savefig(out_path, format=out_path.split('.')[-1], dpi = 100)
	
	# Show the plot and close it after the user is done
	if show: plt.show()
	plt.close()

def plot_compare_images(x_series, shape, title=None, series_labels=None,
	out_path=None, show=True):
	"""
	Construct a grid of grayscale images. It is assumed that the images have
	been normalized (0, 1). The images will be plotted row-wise based off the
	order they were provided.
	
	@param x_series: A sequence of NumPY arrays containing a set of images.
	
	@param shape: The shape of the data. This should be a tuple of x and y
	dimensions.
	
	@param title: The name of the plot.
	
	@param series_labels: The labels for each image in each series. If None, no
	labels will be made.
	
	@param out_path: The full path, including the extension, of where to save
	the plot. If this parameter is None, the image will not be saved.
	
	@param show: If True the image will be shown.
	"""
	
	# Construct the basic plot
	nsamples = 1 if len(x_series[0].shape) == 1 else len(x_series[0])
	nseries = len(x_series)
	fig = plt.figure(figsize=(10, 3))
	gs = gridspec.GridSpec(nseries, nsamples, wspace=0, hspace=0.5)
	if title is not None: fig.suptitle(title)
	if series_labels is None: series_labels = [[None] * nsamples for _ in
		xrange(nseries)]
	
	# Add images
	for i, (x, labels) in enumerate(izip(x_series, series_labels)):
		for j, (xi, y) in enumerate(izip(x, labels)):
			ax = plt.subplot(gs[i, j])
			if y is not None: ax.set_title(y)
			ax.imshow(xi.reshape(shape), cmap=plt.get_cmap('gray'), vmin=0,
				vmax=1, interpolation='none')
			ax.axis('off')
	
	# Save the plot
	if out_path is not None:
		plt.savefig(out_path, format=out_path.split('.')[-1], dpi = 100)
	
	# Show the plot and close it after the user is done
	if show: plt.show()
	plt.close()

###############################################################################
#### 3D Plots
###############################################################################

def plot_surface(x, y, z, x_label=None, y_label=None, z_label=None,
	title=None, out_path=None, show=True, azim=-124, elev=38, vmin=0,
	vmax=100):
	"""
	Basic plotter function for plotting surface plots
	
	@param x: A sequence containing the x-axis data.
	
	@param y: A sequence containing the y-axis data.
	
	@param z: A sequence containing the z-axis data.
	
	@param x_label: The label to use for the x-axis.
	
	@param y_label: The label to use for the y-axis.
	
	@param z_label: The label to use for the z-axis.
	
	@param title: The name of the plot.
	
	@param out_path: The full path to where the image should be saved. The file
	extension of this path will be used as the format type. If this value is
	None then the plot will not be saved, but displayed only.
	
	@param show: If True the plot will be show upon creation.
	
	@param azim: The azimuth to use in the plot.
	
	@param elev: The elevation to use in the plot.
	
	@param vmin: The minimum color value.
	
	@param vmax: The maximum color value.
	"""
	
	# Construct the basic plot
	fig  = plt.figure(figsize=(32, 18), facecolor='w')
	ax   = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(x, y, z, cmap=plt.cm.jet, rstride=1, cstride=1,
		linewidth=0, vmin=vmin, vmax=vmax)
	box = ax.get_position()
	ax2 = plt.axes([(box.x0 + box.width) * 1.0, box.y0, 0.02, box.height])
	plt.colorbar(surf, cax=ax2)
	ax.view_init(azim=azim, elev=elev)
	
	# Add the labels
	if title   is not None : plt.title(title)
	if x_label is not None : ax.set_xlabel(x_label, labelpad=50)
	if y_label is not None : ax.set_ylabel(y_label, labelpad=60)
	if z_label is not None : ax.set_zlabel(z_label, labelpad=20)
	# if x_label is not None : ax.set_xlabel(x_label, labelpad=70)
	# if y_label is not None : ax.set_ylabel(y_label, labelpad=90)
	# if z_label is not None : ax.set_zlabel(z_label, labelpad=30)
	# plt.setp(ax.get_xticklabels(), rotation=5, horizontalalignment='left')
	# plt.setp(ax.get_yticklabels(), rotation=5, horizontalalignment='right')
	# plt.setp(ax.get_zticklabels(), rotation=5, horizontalalignment='right',
		# verticalalignment='center')
	
	# Save the plot
	if out_path is not None:
		plt.savefig(out_path, format=out_path.split('.')[-1], dpi=200)
	
	# Show the plot and close it after the user is done
	if show: plt.show()
	plt.close()

def plot_surface_video(x, y, z, out_dir, x_label=None, y_label=None,
	z_label=None, title=None, elev=20):
	"""
	Basic plotter function for plotting a bunch of views for a surface plot.
	
	@param x: A sequence containing the x-axis data.
	
	@param y: A sequence containing the y-axis data.
	
	@param z: A sequence containing the z-axis data.
	
	@param out_dir: The directory to save the plots in.
	
	@param x_label: The label to use for the x-axis.
	
	@param y_label: The label to use for the y-axis.
	
	@param z_label: The label to use for the z-axis.
	
	@param title: The name of the plot.
	
	@param elev: The elevation to have the 3D plot at.
	"""
	
	# Construct the basic plot
	fig  = plt.figure()
	ax   = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(x, y, z, cmap=plt.cm.jet, rstride=1, cstride=1,
		linewidth=0)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	
	# Add the labels
	if title   is not None : plt.title(title)
	if x_label is not None : ax.set_xlabel(x_label)
	if y_label is not None : ax.set_ylabel(y_label)
	if z_label is not None : ax.set_zlabel(z_label)
	
	# Save the plot
	fig.set_size_inches(19.20, 10.80)
	for i in xrange(360):
		ax.view_init(elev=elev, azim=i)
		plt.savefig(os.path.join(out_dir, '{0}.png'.format(i)), format='png',
			dpi=100)	
	plt.close()