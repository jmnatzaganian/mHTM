# parameter_exploration.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 02/07/16
# 
# Description    : Experiment for studying parameter affects.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Experiment for studying parameter effects.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Native imports
import os, cPickle, time

# Third party imports
import numpy as np

# Program imports
from mHTM.region import SPRegion
from mHTM.datasets.loader import SPDataset
from mHTM.metrics import SPMetrics
from mHTM.plot import plot_line

def main():
	"""
	Program entry.
	
	Build an SP using SPDataset and see how it performs across multiple sets of
	parameters.
	"""
	
	pass

if __name__ == '__main__':
	main()
