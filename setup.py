# setup.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 12/02/15
#	
# Description    : Installs the mHTM project
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

# Native imports
from distutils.core import setup
import shutil

# Install the program
setup(
	name='mHTM',
	version='0.9.0',
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