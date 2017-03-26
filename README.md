# math hierarchical temporal memory (mHTM)

## Introduction
math HTM (mHTM) is a Python build of hierarchical temporal memory (HTM). This
build specifically utilizes the
[cortical learning algorithms (CLA)](|http://numenta.com/assets/pdf/whitepapers/hierarchical-temporal-memory-cortical-learning-algorithm-0.2.1-en.pdf).
A mathematical framework was developed for HTM. This library is built off that
framework. The framework is currently only for the spatial pooler (SP);
however, it is by no means limited to expand. This implementation was
specifically designed to be a completely accurate representation of HTM.
Additionally, care was taken to ensure that efficient computations are
occurring.

Refer to our paper for more details regarding this work.

To aid in tying HTM into the machine learning community, this implementation
was built to be compatible with [Scikit-Learn](http://scikit-learn.org/stable/).
If you are familiar with Scikit-Learn, this API should feel natural.
Additionally, because the Scikit-Learn interface is used, the SP in this
implementation may be used in many of the pre-existing Scikit-Learn tools,
namely those utilizing [cross-validation (CV)](http://scikit-learn.org/stable/modules/cross_validation.html).
A custom [parameter generator](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.ParameterSampler.html)
was constructed, explicitly for that purpose.

The implementation of the SP is single-threaded; however, multiple forms of
parallelizations for cross-validation and parameter optimization exist. For a
local machine, simply using one of Scikit-Learn's CV functions supporting
parallelization should suffice. For a cluster environment, code is provided to
create and launch jobs.

This package is platform independent and should run on any system containing
the prerequisites. If you are using a cluster, the cluster must be running
SLURM, and as such must be a valid Linux distribution.

## Prerequisites
### Required
These prerequisites are needed for working with the base installation:

* [Python 2.7.X](https://www.python.org/downloads/release/python-279/) (all other versions are untested)
* [Numpy](http://www.numpy.org/)
* [matplotlib](http://matplotlib.org/)
* [Scipy](http://www.scipy.org/)
* [Scikit-Learn](http://scikit-learn.org/stable/)
* [Bottleneck](http://berkeleyanalytics.com/bottleneck)

### Optional
These prerequisites are needed by some of the experimental code or for other development purposes:

* [Joblib](https://pythonhosted.org/joblib/index.html)
* [Epydoc](http://sourceforge.net/projects/epydoc/files)
* [graphviz](http://www.graphviz.org/Download..php)

## Installation
### Without virutalenv
1. Install all prerequisites
    1. If you have [pip](https://pip.pypa.io/en/latest/installing.html)
    
            pip install -r requirements.txt
	
    2. If you are on Windows  
        You may download the the (unofficial) precompiled versions available
        from [UCI](http://www.lfd.uci.edu/~gohlke/pythonlibs). Simply download
        the appropriate Python wheel and install it using pip, i.e.
        `pip install my_file.whl`, where `my_file` is the name of your `whl`
        file.

2. Download this repo and execute `python setup.py install`

## Usage
The API is fully documented and available in the "docs" folder. Simply open
"index.html" in your favorite web browser. For convience, click
[here](http://techtorials.me/mHTM/) to browse the docs.

As a starting point, examples have been prepared for working with MNIST. The
dataset is additionally, included. Refer to "mnist_simple.py" in "src/examples"
for a basic introduction into the API. In that same folder, refer to
"mnist_parallel" for examples for using parallelizations locally or an cluster.

The "dev" folder contains the latest code regarding new experiments. This
content is subject to change at any point and is not guaranteed to work;
however, it can also be used as a basis for exploring this library.

## Citing this Work
While this code is completely free of charge, it is highly appreciated that all
uses of this code are cited (both in publications as well as within any
modified code). Note that, this work has been published in
[Frontiers in Robotics and AI](http://journal.frontiersin.org/article/10.3389/frobt.2016.00081/full).  

Mnatzaganian, James, Ernest Fokoué, and Dhireesha Kudithipudi.
"A Mathematical Formalization of Hierarchical Temporal Memory's Spatial
Pooler." Frontiers in Robotics and AI 3 (2016): 81. DOI:
[http://10.3389/frobt.2016.00081](http://10.3389/frobt.2016.00081)

## Bug Reports and Support
No official support is provided; however, support may be provided. To ensure
all feedback is within the same location, please use the Wiki for asking
general questions and create issues if any bugs are found.

## Author
The original author of this code was James Mnatzaganian. For contact info, as
well as other details, see his corresponding [website](http://techtorials.me).

This work was created at RIT's [NanoComputing Research Laboratory](http://www.rit.edu/kgcoe/nanolab/).

## Legal
This code is licensed under the [MIT license](http://opensource.org/licenses/mit-license.php),
with one caveat. Numenta owns patents on specific items. While this code was
written without using any of Numenta's code, it is possible that those patent
laws still apply. Before using this code, commercially, it is recommended to
seek legal advice.