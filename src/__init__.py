# __init__.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 12/02/15
#	
# Description    : Defines the project (main package).
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
math HTM (mHTM) is a Python build of hierarchical temporal memory (HTM). This build specifically utilizes the U{cortical learning algorithms (CLA)<http://numenta.com/assets/pdf/whitepapers/hierarchical-temporal-memory-cortical-learning-algorithm-0.2.1-en.pdf)>}. A mathematical framework was developed for HTM. This library is built off that framework. The framework is currently only for the spatial pooler (SP); however, it is by no means limited to expand. This implementation was specifically designed to be a completely accurate representation of HTM. Additionally, care was taken to ensure that efficient computations are occurring.

We have recently submitted a paper to IEEE TNNLS explaining this work. A preprint is available on U{arXiv<http://arxiv.org/abs/1601.06116>}.

To aid in tying HTM into the machine learning community, this implementation was built to be compatible with U{Scikit-Learn<http://scikit-learn.org/stable/>}. If you are familiar with Scikit-Learn, this API should feel natural. Additionally, because the Scikit-Learn interface is used, the SP in this implementation may be used in many of the pre-existing Scikit-Learn tools, namely those utilizing U{cross-validation (CV)<http://scikit-learn.org/stable/modules/cross_validation.html>}. A custom U{parameter generator<http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.ParameterSampler.html>} was constructed, explicitly for that purpose.

The implementation of the SP is single-threaded; however, multiple forms of parallelizations for cross-validation and parameter optimization exist. For a local machine, simply using one of Scikit-Learn's CV functions supporting parallelization should suffice. For a cluster environment, code is provided to create and launch jobs.

This package is platform independent and should run on any system containing the prerequisites. If you are using a cluster, the cluster must be running SLURM, and as such must be a valid Linux distribution.

Prerequisites
=============
	- U{Python 2.7.X<https://www.python.org/downloads/release/python-279/>}
	- U{Numpy<http://www.numpy.org/>}
	- U{matplotlib<http://matplotlib.org/>}
	- U{Scipy<http://www.scipy.org/>}
	- U{Scikit-Learn<http://scikit-learn.org/stable/>}
	- U{Bottleneck<http://berkeleyanalytics.com/bottleneck>}

Installation
============
	1. Install all prerequisites
		1. If you have U{pip<https://pip.pypa.io/en/latest/installing.html>}
		
			X{pip install numpy matplotlib scipy scikit-learn bottleneck}
		
		2. If you are on Windows
		
			You may download the the (unofficial) precompiled versions available from U{UCI<http://www.lfd.uci.edu/~gohlke/pythonlibs>}. Simply download the appropriate Python wheel and install it using pip, i.e. X{pip install my_file.whl}, where X{my_file} is the name of your X{whl} file.
	2. Download this repo and execute X{python setup.py install}.

Usage
=====
	The API is fully documented and available in the "docs" folder. Simply open "index.html" in your favorite web browser.
	
	As a starting point, examples have been prepared for working with MNIST. The dataset is additionally, included. Refer to "mnist_simple.py" in "src/examples" for a basic introduction into the API. In that same folder, refer to "mnist_parallel" for examples for using parallelizations locally or an cluster.
	
	The "dev" folder contains the latest code regarding new experiments. This content is subject to change at any point and is not guaranteed to work; however, it can also be used as a basis for exploring this library.

Development
===========
	For building the docs, U{Epydoc<http://epydoc.sourceforge.net/>} is used. To generate new docs, 
	
Citing this Work
================
	While this code is completely free of charge, it is highly appreciated that all
	uses of this code are cited (both in publications as well as within any modified code). Once the IEEE TNNLS paper is approved please cite that work. For now, please cite the preprint:
	
	J. Mnatzaganian, E. Fokoue, and D. Kudithipudi, "A Mathematical Formalization of Hierarchical Temporal Memory Cortical Learning Algorithm's Spatial Pooler," arXiv preprint arXiv:1601.06116, 2016.

Bug Reports and Support
=======================
	No official support is provided; however, support may be provided. To ensure all feedback is within the same location, please use the Wiki for asking general questions and create issues if any bugs are found.

Author
======
	The original author of this code was James Mnatzaganian. For contact info, as well as other details, see his corresponding U{website<http://techtorials.me>}.

	This work was created at RIT's U{NanoComputing Research Laboratory<http://www.rit.edu/kgcoe/nanolab/>}.

Legal
=====

	This code is licensed under the U{MIT license<http://opensource.org/licenses/mit-license.php>}, with one caveat. Numenta owns patents on specific items. While this code was written without using any of Numenta's code, it is possible that those patent laws still apply. Before using this code, commercially, it is recommended to seek legal advice.

Connectivity
============
	The following image shows how everything is connected:

	G{importgraph}

Developer Notes
===============
	The following notes are for developers only.

	Installation
	------------
		1.  Download and install U{graphviz<http://www.graphviz.org/Download..
		php>}
		2.  Edit line 95 in X{dev/epydoc_config.txt} to point to the directory
		containing "dot.exe". This is part of the graphviz installation.
		3.  Download this repo and execute X{python setup.py install}.
		4.  Download and install U{Epydoc<http://sourceforge.net/projects/
		epydoc/files>}

	Generating the API
	------------------
		From the mHTM folder, execute X{python epydoc --config=epydoc_config.txt mHTM}

@group Examples: examples
@group Datasets: datasets

@author: U{James Mnatzaganian<http://techtorials.me>}
@organization: U{NanoComputing Research Laboratory<http://www.rit.edu/kgcoe/nanolab/>}
@requires: Python 2.7.X
@version: 0.9.0
@license: U{The MIT License<http://opensource.org/licenses/mit-license.php>}
@copyright: S{copy} 2016 James Mnatzaganian
"""

__docformat__ = 'epytext'