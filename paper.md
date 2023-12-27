---
title: 'EllipticForest: A Direct Solver for Elliptic Partial Differential Equations on Adaptive Meshes'
tags:
  - C++
  - numerical linear algebra
  - partial differential equations
  - adaptive mesh refinement
authors:
  - name: Damyn Chipman
    orcid: 0000-0001-6600-3720
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true
  - name: Donna Calhoun
    orcid: 0000-0002-6005-4575
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
affiliations:
 - name: Boise State University, USA
   index: 1
date: 1 January 2024
bibliography: paper.bib
---

# Summary

EllipticForest is a software library with utilities to solve elliptic partial differential equations (PDEs) with adaptive mesh refinement (AMR) using a direct matrix factorization. It is a quadtree-adaptive implementation of the Hierarchical Poincaré-Steklov (HPS) method @Gillman:2014. The HPS method is a direct method for solving elliptic PDEs based on the recursive merging of Poincaré-Steklov operators @Quarteroni:1991. EllipticForest features coupling with the parallel and highly efficient mesh library `p4est` @Burstedde:2011 for mesh adaptivity and mesh management. Distributed memory parallelism is implemented through the Message Passing Interface (MPI) @Walker:1996, which is a first for the HPS method. The primary feature of EllipticForest is the modularity for users to extend the solver interface to custom solvers at the leaf-level. By default, EllipticForest implements fast cyclic-reduction methods as found in the FISHPACK @Swarztrauber:1999 library and updated in the FISHPACK90 @Adams:2016 code.

Similar to other direct methods, the HPS method is comprised of two stages: a build stage and a solve stage. In the build stage, a set of solution operators are formed that act as the factorization of the system matrix corresponding to the discretization stencil. In the solve stage, the factorization is applied to a right-hand side vector corresponding to boundary and non-homogeneous data to solve the problem with linear complexity $\mathcal{O}(N)$ where $N$ is the size of the system matrix. The advantages of this approach over iterative methods such as conjugate gradient and multi-grid methods include the ability to apply the factorization to multiple right-hand sides.

# Statement of need

The primary novelty of EllipticForest as software is the implementation of the HPS method for user extension. The quadtree-adaptive implementation of the HPS method as featured in [TODO: Cite paper 1] and it's parallel counterpart in [TODO: Cite paper 2] is novel in the field of fast solvers for AMR. This paper highlights the software implementation including  the user-friend interface to the HPS method and the ability for users to extend the solver interface using object-oriented programming (OOP) paradigms. Currently, all other implementations of the HPS method are MATLAB codes that are tailored to each research groups' needs and research endeavors. EllipticForest is the first C++ software ready for user-extension and coupling with other scientific solver libraries.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References