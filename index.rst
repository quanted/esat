.. ESAT documentation master file, created by
   sphinx-quickstart on Wed Dec  6 10:53:28 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EPA's ESAT Documentation
=================================

.. toctree::
   :maxdepth: 3
   :caption: Contents:

The EPA's Environmental Source Apportionment Toolkit (ESAT) project provides an open-source python package with which
source apportionment workflows can be programmatically created and run. The features available in the ESAT package
include data preprocessing and cleaning, non-negative matrix factorization (NMF) algorithms (currently two options are
available), solution analytics and visualizations, error estimation methods, and customized constrained factorization models.
These features reproduce and expand upon what is found in EPA's PMF5 application, which is no longer being updated
or maintained.

ESAT python package focuses on source apportionment estimates using NMF algorithms. These
algorithms are implemented both in python using numpy functions as well as in Rust for an optimization option. The
two currently available are:

| 1. LS-NMF: Least-squares NMF, a well documented and widely uses NMF algorithm. The ls-nmf algorithm is available in the NMF R package.
| 2. WS-NMF: Weight-Semi NMF, a variant of the NMF algorithm which allows for negative values in both the input data matrix and in the factor contribution matrix.

Source apportionment solution error estimation methods are also available, which are the same methods that are found in PMF5.
These are:

| 1. Bootstrap (BS): the input dataset is divided into multiple blocks and randomly reassembled to attempt to quantify the variability in the factor profiles and contributions.
| 2. Displacement (DISP): the solution factor profiles are all individually shifted (both increased and decreased) to determine the amount of change required for the loss value to reach specific dQ (change in Q) values.
| 3. Bootstrap-Displacement (BS-DISP): the combination of the two error estimation methods. Where for each Bootstrap dataset/model, all or targeted factor profile values are adjusted using the DISP method.

Lastly, constrained models are able to be created where the user can define constraints and expressions which set or limit specific factor elements (single values from either the factor profile or factor contribution matrices), or define correlations between factor elements as linear expressions.

Code examples are available in the repository jupyter notebooks, which also provided detailed algorithm documentation and complete workflow examples.


.. toctree::
   :maxdepth: 1
   :caption: Documentation

   self
   README.md
   docs/modules.rst
   docs/cli.rst


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

