# Non-Negative Matrix Factorization
Last Update: 12-11-2023

## Summary

### PMF5
The EPA’s Positive Matrix Factorization (PMF) Model is a mathematical tool that processes a dataset of feature concentrations across many samples (and concentration uncertainties) to estimate a set of sources and their contributions. PMF5 can be used on a wide range of environmental data and is a powerful tool for estimating source apportionment: https://www.epa.gov/air-research/positive-matrix-factorization-model-environmental-data-analyses

PMF version 5 was released in 2014 and is no longer being supported. PMF5 still has an active, international user community; making the development of a modern replacement a worthwhile effort. The primary goals for a PMF5 replacement would be to develop and release an open-source tool that replicates the functionality of PMF5 while also providing new data analytics features and techniques.

### NMF-PY
The focus of the NMF-PY project is to replicate the results and performance of PMF5, using open-source packages and code. Existing mathematics and machine learning packages were investigated for use in the project, though none were able to satisfy the requirements of the project. 

Project Requirements:
1. See high correlations (> 0.95) of the new algorithm’s output to the PMF5 solutions
2. Q of the new algorithms must be less than or comparable to PMF5 (< 10%)
3. Model runtime should be comparable to or faster than PMF5 (< 10%)
4. Use only open-source packages and produce readable code

NMF-PY also tested many different NMF algorithms, since we had to reverse engineer the ME2 solver or develop a comparable process. We have implemented two algorithms that were able to satisfy the constrains of the project.

Additional features will be added to the python package as determined from the user-community and development team.

### Project Products
The products from the NMF-PY project include the following:
1. Open-source python package (with minimal package dependencies)
2. Jupyter Notebooks providing function and algorithm details, as well as complete workflows.
3. Code documentation for users of the python package.
4. Presentations/tutorials on how to use the python package and jupyter notebooks.
5. Manuscript documenting the comparative analysis performed on the output of PMF5 vs the NMF-PY algorithms.
6. Optional: Web/Desktop application wrapper of the python package.

## Requirements
Python = 3.11\
numpy >= 1.24\
pandas >= 2.0.0\
plotly >= 5.14\
scipy >= 1.10

### Rust Requirements
The python package includes a Rust module for running the algorithm update procedures, which requires local compiling to operate.

To run the Rust functions that is specified by the optimized parameter, requires that Rust is installed (https://www.rust-lang.org/tools/install) and the python package maturin (https://pypi.org/project/maturin/) is installed to the py env. 
Then using the py env and from the project root running <i>maturin develop</i> or <i>maturin build</i> to compile the Rust functions to run from the python code. To compile the optimized Rust functions, add the <i>-r</i> or <i>--release</i> tags to compile with optimization.

The rust functions are imported as python functions, under nmf_pyr.

## Creating Docs
The documentation is created using sphinx and several extensions.

To create, update or modify the existing documentation, run 
<i>sphinx-build -M html . docs</i> from the command line at the project root directory running the python environment with the necessary sphinx packages.



