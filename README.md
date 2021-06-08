# Coiling rod

In this repository, you can find the source code used to numerically solve the mathematical problem of a rod coiling about a straight, rigid constraint. The model and the results of the numerical simulations are contained in this paper [[1]](#1).

## Dependencies

The code is written in Python 3 and tested with version 3.8. The following additional libraries are required, in the parentheses we indicate the version used in the simulations reported in the paper:
* FEniCS (version 2019.2.0)
* Numpy (version 1.17.4)
* Scipy (version 1.3.3)

## Repository structure

The repository is organized as follows:
* `continuation.py` contains an implementation of the parameter continuation algorithm; it is part of a bigger library which is currently under development and it is reported here in a simplified version to allow the reproducibility of the results reported in [[1]](#1).
* `problems.py` contains some classes implementing the nonlinear problem of the coiling rod described in the paper for several control parameters and boundary conditions. In particular:
  * `CoilingU2sFree` implements the problem where the natural curvature <img src="https://latex.codecogs.com/svg.latex?u_2^\star" title="u_2^\star" /> is the control parameter and we apply the boundary conditions corresponding to the free ends case (see [[1]](#1)).
  * `CoilingFFree` implements the problem where the force <img src="https://latex.codecogs.com/svg.latex?F" title="-F" /> is the control parameter and we apply the boundary conditions corresponding to the free ends case (see [[1]](#1)).
  * `CoilingU2sFree` implements the problem where the natural curvature <img src="https://latex.codecogs.com/svg.latex?u_2^\star" title="u_2^\star" />  is the control parameter and we apply the boundary conditions corresponding to the pinned ends case (see [[1]](#1)).
  * `CoilingFFree` implements the problem where the force <img src="https://latex.codecogs.com/svg.latex?F" title="-F" />  is the control parameter and we apply the boundary conditions corresponding to the pinned ends case (see [[1]](#1)).
* `example_free.py` solves the problems implemented in the classes `CoilingU2sFree` and `CoilingFFree`, looking for solutions corresponding to helical configurations.
* `example_pinned.py` solves the problems implemented in the classes `CoilingU2sPinned` and `CoilingFPinned`, looking for solutions exhibiting a single perversion.

## Citing

If you find this code useful for your work, please cite [[1]](#1)

## Licence

The source code contained in this repository is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 2.1 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

## References
<a id="1">[1]</a>
D. Riccobelli, G. Noselli, A. DeSimone (2020). Rods coiling about a rigid constraint: Helices and perversions. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering*, 477(2246), 20200817..
