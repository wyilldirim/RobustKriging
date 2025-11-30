# RobustKriging – Robust Variogram Parameter Estimation for Ordinary Kriging

This repository contains MATLAB utilities for robust maximum likelihood
estimation of variogram parameters in Ordinary Kriging, corresponding to the paper:

> **A Robust Maximum Likelihood Estimation Approach for Ordinary Kriging  
> with Outlier-Contaminated Spatial Data**  
> Vural YILDIRIM, Ezio TODINI, Yeliz MERT KANTAR

## Authors and Affiliations

- Vural YILDIRIM  
  Eskisehir Technical University, Institute of Earth and Space Sciences, Eskisehir, Türkiye  
  (vurall_yildirim@hotmail.com)

- Ezio TODINI  
  Italian Hydrological Society, 40127 Bologna, Italy

- Yeliz MERT KANTAR  
  Eskisehir Technical University, Faculty of Science, Department of Statistics, Eskisehir, Türkiye

---

## Current contents

At the moment, the repository provides:

- `KRMLE_functions.m`  
  A MATLAB `classdef` file with **static methods** for:
  - variogram / covariance model evaluation,
  - classical and robust likelihood score and Fisher information,
  - Newton–Raphson estimation of variogram parameters  
    \(nugget, sill, range\) for several models.

No kriging prediction (interpolation) functions are included yet.  
Only **parameter estimation** is implemented here.

---

## Basic usage example

```matlab
% z        : n×1 vector of observations
% H_matrix : n×n distance matrix
% p_initial: [nugget; sill; range] initial guess

v_type         = 'Gau';     % 'Lin','Mono','Gau','Exp','Sph','MSph','Cub'
est_alg        = 'Robust';  % 'NonRobust' or 'Robust'
m_est_func     = 'Tukey';   % see full list in KRMLE_functions header
tol            = 1e-3;      % convergence tolerance
iter_limit     = 100;       % max number of iterations
stop_on_negative = true;    % stop if sill/range stay negative
display_print  = true;      % print iteration progress

[p_final, p_estimates, iter, elapsed_time, converged] = ...
    KRMLE_functions.estimate_variogram_parameters( ...
        z, H_matrix, p_initial, v_type, ...
        est_alg, m_est_func, ...
        tol, iter_limit, stop_on_negative, display_print);

---

## License and usage

At this stage, the code is shared **for peer-review purposes only** in support
of the manuscript:

*A Robust Maximum Likelihood Estimation Approach for Ordinary Kriging with
Outlier-Contaminated Spatial Data*  
Vural YILDIRIM, Ezio TODINI, Yeliz MERT KANTAR

All rights reserved © 2025  
Vural YILDIRIM, Ezio TODINI, Yeliz MERT KANTAR.

No redistribution or re-publication of the source code (in whole or in part)
is permitted without the explicit written permission of the authors.

No public hosting in other repositories or archives is allowed.

No commercial use is allowed.

If you are a reviewer or editor and need additional information or permission,
please contact the corresponding author at:

**vurall_yildirim@hotmail.com**

