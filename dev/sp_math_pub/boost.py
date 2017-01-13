# boost.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 10/12/15
#	
# Description    : Boost experimentations.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Boost experimentations.

G{packagetree mHTM}
"""

# Native imports
from itertools import izip
import os
import cPickle

# Third-Party imports
import numpy as np

# Program imports
from mHTM.plot import plot_surface_video, plot_surface

def compute_boost(duty_cycle, min_duty_cycle, max_boost=10):
	"""
	Evaluate the boost function.
	
	@param duty_cycle: The duty cycle of the current column.
	
	@param min_duty_cycle: The minimum duty cycle for the column's region.
	
	@param max_boost: The max boost.
	"""
	
	if min_duty_cycle == 0:
		return max_boost
	elif duty_cycle > min_duty_cycle:
		return 1
	else:
		return duty_cycle * ((1 - max_boost) / min_duty_cycle) + max_boost

def plot_boost(out_dir, duty_cycle=1000., max_boost=10):
	"""
	Generate some 3D plots for the boost.
	
	@param out_dir: The directory to save the plots in.
	
	@param duty_cycle: The duty cycle to use. This parameter must be a float.
	
	@param max_boost: The max boost to use.
	"""
	
	azim = 73
	elev = 28
	
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	
	data_path = os.path.join(out_dir, 'data.pkl')
	img_path = os.path.join(out_dir, 'boost.png')
	if not os.path.exists(data_path):	
		# Compute the range of values
		duty_cycles = np.arange(0, duty_cycle + 1) / duty_cycle
		min_duty_cycles = np.linspace(0, 1, duty_cycle + 1)
		
		# Make it into a mesh
		x, y = np.meshgrid(duty_cycles, min_duty_cycles)
		
		# Evaluate the boost at each instance
		z = np.array([[compute_boost(xii, yii, max_boost) for xii, yii in
			izip(xi, yi)] for xi, yi in izip(x, y)])
		
		with open(data_path, 'wb') as f:
			cPickle.dump((x, y, z), f, cPickle.HIGHEST_PROTOCOL)
	else:
		with open(data_path, 'rb') as f:
			x, y, z = cPickle.load(f)
	
	# Save the plots
	plot_surface(x, y, z, 'Active Duty Cycle', 'Minimum Active\nDuty Cycle',
		'Boost', None, img_path, False, azim, elev, vmin=None, vmax=None)
	# plot_surface_video(x, y, z, out_dir, 'Duty Cycle', 'Minimum Duty Cycle',
		# 'Boost')

if __name__ == '__main__':
	base_dir = os.path.join(
		os.path.expanduser('~'), 'scratch', 'boost_experiment')
	plot_boost(base_dir, duty_cycle=1000.)
