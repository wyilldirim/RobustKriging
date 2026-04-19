# RobustKriging – Robust Variogram Parameter Estimation for Ordinary Kriging

This repository provides MATLAB implementations of the **Robust Maximum Likelihood Estimation (KRMLE)** framework for variogram parameter estimation in Ordinary Kriging.

The method improves robustness against outliers by integrating M-estimation into the likelihood function, as proposed in the associated publication.

---

## Related publication

This repository accompanies the published paper:

**Yildirim, V., Todini, E., Kantar, Y.M. (2026)**  
*A robust maximum likelihood estimation approach for Ordinary Kriging with outlier-contaminated spatial data*  
Environmental and Ecological Statistics  
https://doi.org/10.1007/s10651-026-00706-9

If you use this code, please cite the paper.

---

## Authors and Affiliations

- **Vural YILDIRIM**  
  Eskisehir Technical University, Institute of Earth and Space Sciences, Eskisehir, Türkiye  
  (vurall_yildirim@hotmail.com)

- **Ezio TODINI**  
  Italian Hydrological Society, 40127 Bologna, Italy

- **Yeliz MERT KANTAR**  
  Eskisehir Technical University, Faculty of Science, Department of Statistics, Eskisehir, Türkiye

---

## Method overview

The KRMLE approach modifies the classical Gaussian log-likelihood used in variogram parameter estimation by introducing robust M-estimation functions.

Instead of the quadratic loss, bounded influence functions (psi-functions) are used to reduce the impact of outliers on parameter estimation.

The method:

- computes standardized residuals,
- applies robust psi-functions (Cauchy, Insha, Logistic),
- solves modified likelihood score equations,
- uses Newton–Raphson or Fisher Scoring for optimization.

This results in stable estimation of variogram parameters even under outlier-contaminated spatial data.

---

## Supported variogram models

The implementation supports:

- Gaussian (`Gau`)
- Exponential (`Exp`)
- Modified Spherical (`MSph`)
- Linear (`Lin`)
- Monomial (`Mono`)

Each model is parameterized by:

- Nugget (`p`)
- Sill (`ω`)
- Range (`α`)

---

## Robust M-estimators

The following robust functions are implemented:

- Cauchy
- Insha
- Logistic

These functions reduce the influence of large residuals and improve robustness in the presence of outliers.

---

## Current contents

The repository currently provides:

- `KRMLE_functions.m`  
  A MATLAB `classdef` file with static methods for:
  - variogram / covariance model evaluation,
  - classical and robust likelihood score functions,
  - Fisher information and Hessian computation,
  - robust parameter estimation for variogram parameters,
  - Newton–Raphson optimization.

The current version focuses on **variogram parameter estimation**.

Future updates may include:

- full Ordinary Kriging prediction,
- spatial interpolation workflows,
- example datasets,
- reproducible benchmark experiments.

---

## Basic usage example

```matlab
% z         : n×1 vector of observations
% H_matrix  : n×n distance matrix
% p_initial : [nugget; sill; range] initial guess

v_type           = 'Gau';      % 'Lin','Mono','Gau','Exp','Sph','MSph','Cub'
est_alg          = 'Robust';   % 'NonRobust' or 'Robust'
m_est_func       = 'Cauchy';   % see full list in KRMLE_functions header
tol              = 1e-3;       % convergence tolerance
iter_limit       = 100;        % maximum number of iterations
stop_on_negative = true;       % stop if sill/range remain negative
display_print    = true;       % print iteration progress

[p_final, p_estimates, iter, elapsed_time, converged] = ...
    KRMLE_functions.estimate_variogram_parameters( ...
        z, H_matrix, p_initial, v_type, ...
        est_alg, m_est_func, ...
        tol, iter_limit, stop_on_negative, display_print);
```

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{yildirim2026krmle,
  title={A robust maximum likelihood estimation approach for Ordinary Kriging with outlier-contaminated spatial data},
  author={Yildirim, Vural and Todini, Ezio and Kantar, Yeliz Mert},
  journal={Environmental and Ecological Statistics},
  year={2026},
  doi={10.1007/s10651-026-00706-9}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions, collaboration, or academic inquiries:

**Vural YILDIRIM**  
Email: vurall_yildirim@hotmail.com
