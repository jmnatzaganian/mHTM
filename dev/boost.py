# boost.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 10/12/15
#	
# Description    : Boost experimentations.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Boost experimentations.

G{packagetree mHTM}
"""

# Native imports
from itertools import izip
import os

# Third-Party imports
import numpy as np

# Program imports
from plot import plot_surface_video, plot_surface

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
	
	# Compute the range of values
	duty_cycles = np.arange(0, duty_cycle + 1) / duty_cycle
	min_duty_cycles = np.linspace(0, 1, duty_cycle + 1)
	
	# Make it into a mesh
	x, y = np.meshgrid(duty_cycles, min_duty_cycles)
	
	# Evaluate the boost at each instance
	z = np.array([[compute_boost(xii, yii, max_boost) for xii, yii in
		izip(xi, yi)] for xi, yi in izip(x, y)])
	
	# Save the plots
	plot_surface(x, y, z, 'Active Duty Cycle', 'Minimum Active\nDuty Cycle',
		'Boost', None, 'boost.png', False)
	# plot_surface_video(x, y, z, out_dir, 'Duty Cycle', 'Minimum Duty Cycle',
		# 'Boost')

if __name__ == '__main__':
	dir = os.path.join('results', 'Boost experiments', '1000_10')
	plot_boost(dir, duty_cycle=1000.)