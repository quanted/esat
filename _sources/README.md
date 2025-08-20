# Environmental Source Apportionment Toolkit (ESAT)
Last Update: 04-16-2025

## Table of Contents
 - [Description](#description)
 - [Quick Start](#quick-start)
   - [Installation](#installation) 
   - [Example Code](#example-code)
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

The ESAT python package has been published with the Journal of Open Source Software (JOSS):

[![DOI](https://joss.theoj.org/papers/10.21105/joss.07316/status.svg)](https://doi.org/10.21105/joss.07316)

## Quick Start

ESAT supports python version 3.10, 3.11, and 3.12. As the python package contains compiled code there are OS and python
specific versions, supporting Mac Intel, Mac M1+, Linux, and Windows.

### Documentation
The Python API and CLI documentation can be found at the GitHub ESAT IO site: [https://quanted.github.io/esat/](https://quanted.github.io/esat/)

### Installation
The ESAT python package contains all compiled code and required dependencies and can be installed using pip
```bash
pip install esat
```
which will install the latest version that supports and is available for your python version and OS.

Development versions of ESAT can be found on the GitHub actions page, for logged-in users, 
under the 'Build and Publish Wheel' workflow. The latest version of the package will be available as an artifact for 
download in the 'Artifacts' section of the completed workflow. There wheel files can be found for specific versions
of python and supported operating systems. 

If an error message appears during installation stating that the 
package is not supported check that the correct OS and python version are being installed for that system. The python 
wheels can be installed directly using 
```bash
pip install <wheel file name>
```
The esat python package is recommended to be installed in its own dedicated python virtual environment or conda environment.

To run the jupyter notebooks, install jupyterlab into the esat python environmental
```bash
pip install jupyterlab
```

### Example Code
Jupyter notebooks containing complete code examples, using sample datasets, are available for the 
[source apportionment workflow](notebooks/epa_esat_workflow_01.ipynb) and the [simulator workflow](notebooks/epa_esat_simulator_01.ipynb). 

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

### Limitations
Matrix factorization algorithms are fall under the group of optimization, or minimization, algorithms which attempt to 
find a minima based upon some loss function and stopping condition. These algorithms are classified as NP-Complete, a 
category of algorithms which are nondeterministic polynomial time complete and there is no known way to quickly find a 
solution. Given this limitation of NMF, a solution can only be considered a local minima with no known way to guarantee 
or prove it is the globally optimal solution. One approach for helping determine that a solution is a good solution is 
by producing many such solutions with a constricted convergence criteria or stopping condition. Then evaluating these 
solutions to determine which, if any, correspond to the best actual representation or model of the data given domain 
knowledge and expertise. 

NMF algorithms are data-agnostic, operates the same on any correctly structured data regardless of domain, potential 
leading to another limitation, interpretation of the solution. An important component in evaluating whether or not to 
use ESAT, or any NMF algorithm, on a dataset is to determine how to interpret factor profiles and contributions. How 
this is done is fully dependent on the data and domain of the input dataset, such as units, types of features, temporal 
or spatial considerations, etc. 


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
 * Support can be requested through GitHub issues or through email at [esat@epa.gov](mailto:esat@epa.gov).

## Disclaimer 
ESAT development has been funded by U.S. EPA.  Mention of any trade names, products, or services does not convey, and 
should not be interpreted as conveying, official EPA approval, endorsement, or recommendation. The views expressed in 
this README are those of the authors and do not necessarily represent the views or policies of the US EPA.