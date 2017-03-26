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
from distutils.sysconfig import get_python_lib
import shutil, os

# Remove any old versions
print 'Removing old versions...'
py_libs = get_python_lib()
for path in os.listdir(py_libs):
	if path[:4] == 'mHTM':
		full_path = os.path.join(py_libs, path)
		if os.path.isfile(full_path):
			try:
				os.remove(full_path)
			except OSError:
				pass
		else: shutil.rmtree(full_path, True)

# Install the program
print 'Installing...'
setup(
	name='mHTM',
	version='0.11.1',
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
print 'Cleaning up...'
shutil.rmtree('build', True)
