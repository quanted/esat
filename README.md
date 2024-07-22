# Environmental Source Apportionment Toolkit (ESAT)
Last Update: 07-22-2024

## Summary

### PMF5
The EPA’s Positive Matrix Factorization (PMF) Model is a mathematical tool that processes a dataset of feature concentrations across many samples (and concentration uncertainties) to estimate a set of sources and their contributions. PMF5 can be used on a wide range of environmental data and is a powerful tool for estimating source apportionment: https://www.epa.gov/air-research/positive-matrix-factorization-model-environmental-data-analyses

PMF version 5 was released in 2014 and is no longer being supported. PMF5 still has an active, international user community; making the development of a modern replacement a worthwhile effort. The primary goals for a PMF5 replacement would be to develop and release an open-source tool that replicates the functionality of PMF5 while also providing new data analytics features and techniques.

### ESAT
The EPA's Environmental Source Apportionment Toolkit (ESAT) project provides an open-source python package with which
source apportionment workflows can be programmatically created and run. The features available in the ESAT package
include data preprocessing and cleaning, non-negative matrix factorization (NMF) algorithms (currently two options are
available), solution analytics and visualizations, error estimation methods, and customized constrained factorization models.
These features reproduce and expand upon what is found in EPA's PMF5 application, which is no longer being updated
or maintained.

ESAT python package focuses on source apportionment estimates using NMF algorithms. These
algorithms are implemented both in python using numpy functions as well as in Rust for an optimization option. The
two currently available are:
 1. LS-NMF: Least-squares NMF, a well documented and widely uses NMF algorithm. The ls-nmf algorithm is available in the NMF R package.

 2. WS-NMF: Weight-Semi NMF, a variant of the NMF algorithm which allows for negative values in both the input data matrix and in the factor contribution matrix.

Source apportionment solution error estimation methods are also available, which are the same methods that are found in PMF5.
These are:
 1. Bootstrap (BS): the input dataset is divided into multiple blocks and randomly reassembled to attempt to quantify the variability in the factor profiles and contributions.

 2. Displacement (DISP): the solution factor profiles are all individually shifted (both increased and decreased) to determine the amount of change required for the loss value to reach specific dQ (change in Q) values.

 3. Bootstrap-Displacement (BS-DISP): the combination of the two error estimation methods. Where for each Bootstrap dataset/model, all or targeted factor profile values are adjusted using the DISP method.

Lastly, constrained models are able to be created where the user can define constraints and expressions which set or limit specific factor elements (single values from either the factor profile or factor contribution matrices), or define correlations between factor elements as linear expressions.

### Notebooks
Code examples are available in the repository jupyter notebooks, which also provided detailed algorithm documentation and complete workflow examples.

## Requirements
Code has been developed and tested using Python 3.12

Code runtime requirements can be found and installed using the requirements.txt file.

Development environments requirements, for running jupyter notebooks, compiling Rust code, and running Sphinx for documentation can be in the requirements-dev.txt file.

### Rust Requirements
The python package includes a Rust module for running the algorithm update procedures, which requires local compiling to execute.

To run the Rust functions that is specified by the optimized parameter, requires that Rust is installed (https://www.rust-lang.org/tools/install) and the python package maturin (https://pypi.org/project/maturin/) is installed to the python development environment. 
Then from the python env and the project root, executing <i>maturin develop</i> will compile the Rust code and place it in your python environment path. To compile optimized Rust code, include the <i>-r</i> or <i>--release</i> tags.

The Rust code can also be compiled to the target directory inside of project root using <i>maturin build</i>.

The rust functions are imported as python functions, with the 'from esat_rust import esat_rust'

### Creating Docs
The documentation is created using sphinx and several extensions.

If creating or adding new rst files run <i>sphinx-apidoc -o docs esat</i>.

To create, update or modify the existing documentation html, run 
<i>sphinx-build -M html . docs</i> from the command line at the project root directory running the python environment with the necessary sphinx packages.

### Building Wheels and Compiling Rust
The ESAT python package and cli are built using setuptools and setuptools-rust, with configuration details defined in pyproject.toml and Cargo.toml. 

The python package can be built with the standard <i>python -m build</i> from the project root directory. 

Build will compile the rust code and package up the python
code combining them into the wheel for distribution. The resulting wheel and compressed file will be found in the project dist directory.
