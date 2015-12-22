# setup.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
# Date Created   : 12/02/15
#	
# Description    : Installs the mHTM project
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

# Native imports
from distutils.core import setup
import shutil

# Install the program
setup(
	name='mHTM',
	version='1.0.0',
	description="HTM CLA Implementation",
	author='James Mnatzaganian',
	author_email='jamesmnatzaganian@outlook.com',
	url='http://techtorials.me',
	package_dir={'mHTM':'src', 'mHTM.datasets':'src/datasets',
		'mHTM.examples':'src/examples'},
	packages=['mHTM', 'mHTM.datasets', 'mHTM.examples'],
	package_data={'mHTM.datasets':['mnist.pkl']}
	)

# Remove the unnecessary build folder
try:
	shutil.rmtree('build')
except:
	pass