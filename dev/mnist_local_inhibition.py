# mnist_local_inhibition.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# Website        : https://www.rit.edu/kgcoe/nanolab/
# Date Created   : 12/08/15
# 
# Description    : Run the local inhibition experiments for MNIST.
# Python Version : 2.7.X
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2016 James Mnatzaganian

"""
Run the local inhibition experiments for MNIST.

G{packagetree mHTM}
"""

__docformat__ = 'epytext'

# Third party imports
import numpy as np

# Program imports
from mHTM.examples.mnist_parallel import main_slurm

def main():
	seed = 123456789
	np.random.seed(seed)
	main_slurm(
		log_dir='results/partial_mnist-local-fixed',
		ntrain=800,
		ntest=200,
		niter=1000,
		nsplits=5,
		global_inhibition=False,
		partition_name='work',
		seed=seed
	)

if __name__ == '__main__':
	main()
