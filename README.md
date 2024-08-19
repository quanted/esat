# Environmental Source Apportionment Toolkit (ESAT)
Last Update: 08-15-2024

## Table of Contents
 - [Description](#description)
 - [Background](#background)
    - [ESAT Predecessor](#esat-predecessor)
    - [Features](#features)
    - [Notebooks](#notebooks)
 - [Development](#development)
    - [Requirements](#requirements)
    - [Rust Compiling](#rust-compiling)
    - [Creating Docs](#creating-docs)
    - [Building Wheels and Compiling Rust](#building-wheels-and-compiling-rust)
    - [Code Tests](#code-tests)
    - [Community Contributions](#community-contributions)

## Description
The Environmental Source Apportionment Toolkit (ESAT) is an open-source software package that provides API and CLI 
functionality to create source apportionment workflows specifically targeting environmental datasets. Source 
apportionment in environment science is the process of mathematically estimating the profiles and contributions of 
multiple sources in some dataset, and in the case of ESAT, while considering data uncertainty. There are many potential 
use cases for source apportionment in environmental science research, such as in the fields of air quality, water 
quality and potentially many others.

The ESAT toolkit is written in Python and Rust, and uses common packages such as numpy, scipy and pandas for data 
processing. The source apportionment algorithms provided in ESAT include two variants of non-negative matrix 
factorization (NMF), both of which have been written in Rust and contained within the python package. A collection of 
data processing and visualization features are included for data and model analytics. The ESAT package includes a 
synthetic data generator and comparison tools to evaluate ESAT model outputs.

## Background

### ESAT Predecessor
A widely used application used for environmental source apportionment is the EPA's Positive Matrix Factorization version
5 (PMF5), which is a broadly used tool with an international user community. The PMF5 application is a mathematical tool
that processes a dataset of feature concentrations across many samples (and concentration uncertainties) to estimate a 
set of source profiles and their contributions. PMF5 can be used on a wide range of environmental data and is a powerful
tool for estimating source apportionment: 
https://www.epa.gov/air-research/positive-matrix-factorization-model-environmental-data-analyses

PMF5 was released in 2014 and is no longer being supported. The math engine used in PMF5 is proprietary and the source
code has not been made public. One of the primary purposes of ESAT was to recreate the source apportionment workflow and
mathematics as an open-source software package to offer a modernized option for environmental source apportionment. 
Other reasons for developing ESAT was to offer increased maintainability, development efficient, thorough documentation,
modern optimizations, new features and customized workflows for novel use cases.

### Features
ESAT python package focuses on source apportionment estimates using NMF algorithms. These
algorithms are implemented both in python using numpy functions and in Rust (default) for an optimization option. The
two currently available are:
 1. LS-NMF: Least-squares NMF, a well documented and widely uses NMF algorithm. The ls-nmf algorithm is available in the NMF R package.

 2. WS-NMF: Weight-Semi NMF, a variant of the NMF algorithm which allows for negative values in both the input data matrix and in the factor contribution matrix.

Source apportionment solution error estimation methods are also available, which are the same methods that are found in PMF5.
These are:
 1. Bootstrap (BS): the input dataset is divided into multiple blocks and randomly reassembled to attempt to quantify the variability in the factor profiles and contributions.

 2. Displacement (DISP): the solution factor profiles are all individually shifted (both increased and decreased) to determine the amount of change required for the loss value to reach specific dQ (change in Q) values.

 3. Bootstrap-Displacement (BS-DISP): the combination of the two error estimation methods. Where for each Bootstrap dataset/model, all or targeted factor profile values are adjusted using the DISP method.

ESAT includes constrained models, as found in PMF5, where selecting a source apportionment model there is the option to add constraints through defining specific value constraints or define value correlations as a collection of linear equations.

Lastly, ESAT includes a data simulator which allows for random or use defined synthetic source profiles and contributions to be used in ESAT to evaluating how well the original synthetic data can be recreated.

### Notebooks
Juypter notebooks are available that demonstrate the complete source apportionment and error estimation workflow found in PMF5, demonstrated in notebooks/epa_esat_workflow_01.ipynb

The simulator notebook provides examples for creating the synthetic profiles and contributions dataset and using the evaluation features to see how 'well' ESAT can recreate those profiles and contributions.

Other notebooks are included which were used during development and verifying visualizations. 

## Development
### Requirements
* Core ESAT python package requirements can be found in the requirements.txt file.
* The python requirements for creating the code documentation can be found in the doc-requirements.txt file.
* Full development python package requirements can be found in the _dev-requirements.txt file (not actively maintained).

The ESAT python codebase includes github workflow actions which run:
1. Run python build to compile the Rust code and create python packages for python 3.10, 3.11 and 3.12 on Linux, Windows and MacOS.
2. Recreate code documentation from the README.md file, code docstrings for the Python API and CLI. Documentation is used to update the github documentation site for ESAT.

### Rust Compiling
The python package includes a Rust module for running the algorithm update procedures, which requires local compiling to execute.

To run the Rust functions that is specified by the optimized parameter, requires that Rust is installed (https://www.rust-lang.org/tools/install) and the python package maturin (https://pypi.org/project/maturin/) is installed to the python development environment. 
Then from the python env and the project root, executing <i>maturin develop</i> will compile the Rust code and place it in your python environment path. To compile optimized Rust code, include the <i>-r</i> or <i>--release</i> tags.

The Rust code can also be compiled to the target directory inside of project root using <i>maturin build</i>.

The rust functions are imported as python functions, with the 'from esat_rust import esat_rust'.

When creating the python package the pyproject.toml specifies that both setuptools and setuptool-rust are used. Setuptools-rust is required for compiling the Rust code during package build. 

### Creating Docs
The documentation is created using sphinx and several extensions.

To create or add new rst files run <i>sphinx-apidoc -o docs esat</i>.

To create, update or modify the existing documentation html, run 
<i>sphinx-build -M html . docs</i> from the command line at the project root directory running the python environment with the necessary sphinx packages.

### Building Wheels and Compiling Rust
The ESAT python package and cli are built using setuptools and setuptools-rust, with configuration details defined in pyproject.toml and Cargo.toml. 

The python package can be built with the standard <i>python -m build</i> from the project root directory. 

Build will compile the rust code and package up the python
code combining them into the wheel for distribution. The resulting wheel with the compiled code is available on github as a workflow artifact for the targeted architecture and python version.

The python package will be available on pypi.org in the near future.

### Code Tests
A collection of pytest tests have been created to test functionality, mostly as systems tests, which can be executed as 

<i>coverage run -m pytest tests</i>

with the coverage results displayed by 

<i>coverage report</i>

While the overall coverage percentage is low, the majority of the untested code is for visualization functions with all 
core functionality covered by tests.

### Community Contributions
For those in the user community wishing to contribute to this project:
 * Code updates can be made through pull requests that will be reviewed by repository maintainers.
 * Software, code, or algorithm related bugs and issues can be submitted directly as issues on the GitHub repository.
 * Support can be requested through GitHub issues or through email at [esat@epa.gov](esat@epa.gov).

