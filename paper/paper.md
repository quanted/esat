---
title: 'ESAT: Environmental Source Apportionment Toolkit Python package'
tags: 
    - Python
    - source apportionment
    - matrix factorization
    - environmental research
    - air quality
    - water quality
authors:
    - name: Deron Smith
      orcid: 0009-0009-4015-5270
      affiliation: 1
    - name: Michael Cyterski
      orcid: 0000-0002-8630-873X
      affiliation: 1
    - name: John M Johnston
      orcid: 0000-0002-5886-7876
      affiliation: 1
    - name: Kurt Wolfe
      orcid: 
      affiliation: 1
    - name: Rajbir Parmar
      orcid: 0009-0005-2221-0433
      affiliation: 1
affiliations:
    - name: United States Environmental Protection Agency, Office of Research and Development, Center for Environmental Measurement and Modeling
      index: 1
date: TBD
bibliography: paper.bib
---

# Summary

Source apportionment is an important tool in environmental science where sample or sensor data are often the product
of many, often unknown, contributing sources. One technique for source apportionment is non-negative matrix 
factorization (NMF). Using NMF, source apportionment models estimate potential source profiles and contributions providing 
a cost-efficient method for further strategic data collection or modeling. An important aspect of modeling, especially 
environmental modeling, is the consideration of input data uncertainty and error quantification. 

The EPA's Positive Matrix Factorization version 5 (PMF5)[@PMF5] application offers a source apportionment modeling and analysis
workflow that has an active international user community. PMF5 was released in 2014 and is no longer supported; 
additionally the Multilinear Engine v2 (ME2) used in PMF5 is proprietary, with documentation existing only for the prior version ME1
[@Paatero:1999].  

# Statement of Need

The Environmental Source Apportionment Toolkit (ESAT) has been developed as a replacement to PMF5, and has been 
designed for increased flexibility, documentation and transparency.`ESAT` is an open-source Python 
package for flexible source apportionment workflows. The Python API and CLI of `ESAT` provides an object-oriented 
interface that can completely recreate the PMF5 workflow. The matrix factorization algorithms in `ESAT` have been 
written in Rust for optimization of the core math functionality. `ESAT` has two NMF algorithms for updating
the profile and contribution matrices of the solution: least-squares NMF (LS-NMF) [@Wang:2006] and weighted-semi NMF 
(WS-NMF) [@Ding:2008] [@DeMelo:2012]. 

`ESAT` provides a highly flexible API and CLI that can create source apportionment workflows like those found in PMF5, 
but can also create new workflows that allow for novel environmental research. 
`ESAT` was developed for environmental research, though it's not limited to that domain, as matrix
factorization is used in many different fields; `ESAT` places no restriction on the types of input datasets.

## Algorithms
The loss function used in `ESAT`, and PMF5, is a variation of squared-error loss, where data uncertainty is taken into
consideration:

$$ 
Q = \sum_{i=1}^n \sum_{j=1}^m \bigg[ \frac{V_{ij} - \sum_{k=1}^K W_{ik} H_{kj}}{U_{ij}} \bigg]^2 
$$
here $V$ is the input data matrix of features (columns=$M$) by samples (rows=$N$), $U$ is the uncertainty matrix of the 
input data matrix, $W$ is the factor contribution matrix of samples by factors=$k$, $H$ is the factor profile of 
factors by features.

The `ESAT` versions of NMF algorithms convert the uncertainty $U$ into weights defined as $Uw = \frac{1}{U^2}$. 
The update equations for LS-NMF then become:

$$ H_{t+1} = H_t \circ \frac{W_t (V \circ Uw)}{W_t {((W_{t}H_{t}) \circ Uw)}} $$

$$ W_{t+1} = W_t \circ \frac{(V \circ Uw) H_{t+1}}{((W_{t}H_{t+1})\circ Uw )H_{t+1}} $$

while the update equations for WS-NMF:

$$ W_{t+1,i} = (H^{T}Uw_{i}^{d}H)^{-1}(H^{T}Uw_{i}^{d}V_{i})$$

$$ H_{t+1,i} = H_{t, i}\sqrt{\frac{((V^{T}Uw)W_{t+1})_{i}^{+} + [H_{t}(W_{t+1}^{T}Uw W)^{-}]_{i}}{((V^{T}Uw)W_{t+1})_{i}^{-} + [H_{t}(W_{t+1}^{T}Uw W)^{+}]_{i}}}$$

where $W^{-} = \frac{(|W| - W)}{2.0}$ and $W^{+} = \frac{(|W| + W)}{2.0}$.

## Error Estimation
An important part of the source apportionment workflow is quantifying potential model error. `ESAT` offers the same error estimation
methods that were available in PMF5 [@Brown:2015], but with flexibility for customization.
\begin{itemize}
    \item Displacement Method (DISP): Quantify the error due to rotational ambiguity by evaluating the amount of change in source profile that correspond to specific changes in the loss. 
    \item Bootstrap Method (BS): Quantify the error due to the order of the samples via block resampling. 
    \item BS-DISP: Calculate the displacement error on a set of bootstrap datasets to quantify the combined error.
\end{itemize}

## Simulator
`ESAT` contains a data simulator for generating synthetic profiles and contributions which allow for direct model evaluation. 
The synthetic profiles can either be randomly generated, use a previously defined set of profiles, or a combination of both. 
The random synthetic contributions can follow specified curves and value ranges. The `ESAT` model profiles can then 
be mapped to the known synthetic data for direct comparison and accuracy evaluations. 

# Acknowledgements
We thank Tom Purucker, and Jeffery Minucci for manuscript and code review and edits. 
This paper has been reviewed in accordance with EPA policy and approved for publication. 
ESAT development has been funded by U.S. EPA.  Mention of any trade names, products, or services does not convey, and 
should not be interpreted as conveying, official EPA approval, endorsement, or recommendation. The views expressed in 
this paper are those of the authors and do not necessarily represent the views or policies of the US EPA.

# References