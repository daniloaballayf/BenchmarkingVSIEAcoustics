# Acoustic Volume-Surface Integral Equation Benchmarks

> This repository provides four benchmarks to verify the acoustic VSIE for nested geometries against spherical harmonics, BEM, and FEM-BEM algorithms.

---

## Overview

This repository contains Python code for the Volume-Surface Integral Equation (VSIE) for acoustic wave propagation through nested geometries.
The material parameters can be heterogeneous in the interior and have jumps across interfaces.
The computational results are compared against reference solutions obtained from spherical harmonics, Boundary Element Method (BEM), and Finite Element Method - Boundary Element Method (FEM-BEM) algorithms.

This repository accompanies the manuscript "*Benchmarking Nested Volume-Surface Integral Equations for Acoustics*."
A preprint can be found on arXiv.

---

## Installation

The repository requires a standard Python environment with the following libraries.
- `vsie`: the source code of the VSIE algorithm is provided in the `vsie` folder in this repository.
- `bempp-cl`: for the BEM reference solutions. The benchmarks were performed with version 0.3.1.
- `fenics-dolfinx`: for the FEM-BEM reference solutions. The benchmarks were performed with version 0.6.0.

The Python environment can be set up using Conda as follows.

```bash
conda create -n vsie-env
conda activate vsie-env
conda install bempp-cl=0.3.1 fenics-dolfinx=0.6.0 "setuptools<82" jupyter numpy scipy numba gmsh meshio matplotlib mpich
```

---

## Usage

Each folder contains a Jupyter notebook `Results.ipynb` that demonstrates how to run the benchmark and compare the results with the reference solution.
Calculated results for compute-intensive simulations are provided in the data folders.

---

## Acknowledgements

- This work was financially supported by the Agencia Nacional de Investigación y Desarrollo (ANID), Chile [FONDECYT 1230642].
- Compute resources were provided by the Pontificia Universidad Católica de Chile.
- Some routines for the postprocessing of the VSIE results were adapted from the `optimus` library ([@optimuslib](https://github.com/optimuslib/optimus)).

---

## Authors

- Danilo Aballay ([@daniloaballayf](https://github.com/daniloaballayf))
- Elwin van 't Wout ([@evantwout](https://github.com/evantwout))

---
