% Copyright (c) 2025
% Vural YILDIRIM, Ezio TODINI, Yeliz MERT KANTAR
% All rights reserved.
% This code is shared for peer-review purposes only and may not be
% redistributed or used commercially without the prior written
% permission of the authors.

classdef KRMLE_functions
     %KRMLE_FUNCTIONS
    %   Robust variogram parameter estimation utilities for the paper:
    %
    %   "A Robust Maximum Likelihood Estimation Approach for Ordinary
    %    Kriging with Outlier-Contaminated Spatial Data"
    %
    %   Authors:
    %       Vural YILDIRIM
    %       Ezio TODINI
    %       Yeliz MERT KANTAR
    %
    %   Affiliations:
    %       a  Eskisehir Technical University, Institute of Earth and Space Sciences,
    %          Eskisehir, Türkiye (vurall_yildirim@hotmail.com)
    %       b  Italian Hydrological Society, 40127 Bologna, Italy
    %       c  Eskisehir Technical University, Faculty of Science,
    %          Department of Statistics, Eskisehir, Türkiye
    %
    %   Repository:
    %       https://github.com/wyilldirim/RobustKriging
    %
    %   -------------------------------------------------------------------
    %   EXAMPLE CALL
    %   -------------------------------------------------------------------
    %   Robust Gaussian variogram with Tukey M-estimator:
    %
    %       v_type        = 'Gau';        % Gaussian variogram
    %       est_alg       = 'Robust';     % 'NonRobust' or 'Robust'
    %       m_est_func    = 'Tukey';      % one of the M-estimators listed below
    %       tol           = 1e-3;         % convergence tolerance
    %       iter_limit    = 100;          % maximum number of iterations
    %       stop_on_negative = true;      % stop if sill/range stay negative
    %       display_print = true;         % print iteration progress
    %
    %       [p_final, p_estimates, iter, elapsed_time, converged] = ...
    %           KRMLE_functions.estimate_variogram_parameters( ...
    %               z, H_matrix, p_initial, v_type, ...
    %               est_alg, m_est_func, ...
    %               tol, iter_limit, stop_on_negative, display_print);
    %
    %   -------------------------------------------------------------------
    %   OPTION DOMAINS (ALLOWED VALUES)
    %   -------------------------------------------------------------------
    %
    %   v_type  : variogram / covariance model
    %       v_type ∈ {
    %           'Lin',   ... % Linear
    %           'Mono',  ... % Monomial
    %           'Gau',   ... % Gaussian
    %           'Exp',   ... % Exponential
    %           'Sph',   ... % Spherical
    %           'MSph',  ... % Modified Spherical
    %           'Cub'    ... % Cubic
    %       }
    %
    %   est_alg : estimation algorithm for the likelihood
    %       est_alg ∈ {
    %           'NonRobust', ... % classical Gaussian likelihood
    %           'Robust'        % M-estimation based robust likelihood
    %       }
    %
    %   m_est_func : robust M-estimator name (case-sensitive!)
    %       m_est_func ∈ {
    %           'OLS',           ...
    %           'LAD',           ...
    %           'L1L2',          ...
    %           'Fair',          ...
    %           'Cauchy',        ...
    %           'Geman_McClure', ...
    %           'Welsch',        ...
    %           'Huber',         ...
    %           'Tukey',         ...
    %           'Bell',          ...
    %           'Bell2',         ...
    %           'Logistic',      ...
    %           'insha',         ...  % note: lower-case 'insha'
    %           'Andrews',       ...
    %           'Hampel',        ...
    %           'Bisquare'
    %       }
    %
    %   tol : convergence tolerance
    %       - positive scalar, e.g.
    %           tol = 1e-3;    % loose / fast
    %           tol = 1e-4;    % tighter / slower
    %
    %   iter_limit : maximum number of Newton iterations
    %       - positive integer, e.g.
    %           iter_limit = 50;
    %           iter_limit = 100;
    %
    %   stop_on_negative : early stop if sill or range stay negative
    %       - logical:
    %           true  → stop if sill/range are negative for 10 consecutive iterations
    %           false → continue, only print warnings (if display_print = true)
    %
    %   display_print : verbosity flag
    %       - logical:
    %           true  → print iteration info, parameter values, convergence status
    %           false → fully silent (only outputs are returned)
    %
    %   NOTE:
    %       All string options are **case-sensitive** and must match exactly
    %       the names implemented in:
    %           - robust_m_estimator
    %           - calculate_variogram_variance_and_derivatives
    %
    %   -------------------------------------------------------------------
    %   This class only contains parameter-estimation utilities.
    %   No kriging prediction (interpolation) functions are defined here.
    %   -------------------------------------------------------------------
    %

    methods (Static)        

        % ------------------------------------------------------------------
        % 1) VARIOGRAM PARAMETER ESTIMATION
        % ------------------------------------------------------------------
        %
        %   This routine performs iterative estimation of the variogram
        %   (covariance) parameters p = [nugget, sill, range] using a
        %   Newton–Raphson type update driven by:
        %       - the score vector (classical or robust),
        %       - and the corresponding Fisher information matrix.
        %
        %   The algorithm supports:
        %       - NonRobust : Gaussian log-likelihood,
        %       - Robust    : M-estimation based modifications of the score
        %                    and Fisher information, using standardized
        %                    residuals and a chosen psi-function.
        %
        %   INPUTS
        %     z              : n×1 data vector (observations)
        %     H_matrix       : n×n distance matrix
        %     p_ini          : 3×1 initial parameter vector [nugget; sill; range]
        %     v_type         : variogram model type (e.g. 'Gau','Exp','MSph', ...)
        %     est_alg        : 'NonRobust' or 'Robust' (default: 'NonRobust')
        %     m_est_func     : robust M-estimator name
        %                      (e.g. 'insha','Tukey','Cauchy', ...), used
        %                      only if est_alg = 'Robust'
        %     tol            : convergence tolerance on max|p_new - p_old|
        %                      (default: 1e-3)
        %     iter_limit     : maximum number of iterations allowed
        %                      (default: 100)
        %     stop_on_negative : logical
        %         true  → stop if sill/range negative for 10 consecutive iterations
        %         false → continue even with negative values (only log)
        %     display_print  : logical
        %         true  → print iteration info to screen
        %         false → silent mode (no output)
        %
        %   OUTPUTS
        %     p_final      : 3×1 final parameter vector
        %     p_estimates  : 3×K matrix, each column is p at an iteration
        %     iter         : number of iterations performed
        %     elapsed_time : total runtime in seconds
        %     converged    : logical flag (true if max|Δp| ≤ tol, false otherwise)
        %

        function [p_final, p_estimates, iter, elapsed_time, converged] = ...
            estimate_variogram_parameters(z, H_matrix, p_ini, v_type, est_alg, m_est_func, tol, iter_limit, stop_on_negative, display_print)

            % ---------- defaults ----------
            if nargin < 5 || isempty(est_alg)
                est_alg = 'NonRobust';
            end
            if nargin < 6 || isempty(m_est_func)
                m_est_func = 'None';
            end
            if nargin < 7 || isempty(tol)
                tol = 1e-3;
            end
            if nargin < 8 || isempty(iter_limit)
                iter_limit = 100;
            end
            if nargin < 9 || isempty(stop_on_negative)
                stop_on_negative = false;
            end
            if nargin < 10 || isempty(display_print)
                display_print = false;  % Default: silent mode
            end
        
            % Ensure column vector for initial parameters
            p_ini = p_ini(:);
        
            % ---------- initialization ----------
            p_estimates       = [];    % will collect p at each iteration (columns)
            fark              = Inf;   % difference for convergence check
            iter              = 0;     % iteration counter
            neg_count         = 0;     % consecutive negative sill/range counter
            aborted_negative  = false; % early termination flag
        
            % p_new undefined prevention - set to p_ini initially
            p_new = p_ini;
        
            tic;  % start timer
        
            % ---------- Newton–Raphson loop ----------
            while (fark > tol) && (iter < iter_limit)
        
                if iter == 0
                    % First iteration: start from p_ini
                    p_old      = p_ini;
                    p_estimates = [p_estimates, p_old];
                else
                    % Otherwise: use previous p_new
                    p_old = p_new;
                end
        
                % ----- scoring and Fisher information -----
                switch est_alg
                    case 'NonRobust'
                        % Center data by the empirical mean
                        z_ = z - mean(z);
                        scoring = KRMLE_functions.calculate_scoring(z_, H_matrix, p_old, v_type);
                        fisher  = KRMLE_functions.calculate_fisher(z_, H_matrix, p_old, v_type);
        
                    case 'Robust'
                        % Trim top/bottom 10% to obtain a robust location
                        z_ = z - trimmean(z, 10);
                        scoring = KRMLE_functions.calculate_robust_scoring(z_, H_matrix, p_old, m_est_func, v_type);
                        fisher  = KRMLE_functions.calculate_robust_fisher(z_, H_matrix, p_old, m_est_func, v_type);
        
                    otherwise
                        error('est_alg must be ''NonRobust'' or ''Robust''.');
                end
        
                % Numerical stabilisation of Fisher matrix
                fisher = fisher + 1e-4 * eye(size(fisher));
        
                % ----- Newton–Raphson update -----
                % p_new = p_old - step * fisher^{-1} * scoring
                step_factor = 0.5;
                p_new = p_old - step_factor * (fisher \ scoring);
        
                % Optional: enforce nugget >= 0
                % if p_new(1) < 0
                %     p_new(1) = 0;
                % end
        
                % --- check if sill or range negative ---
                if p_new(2) < 0 || p_new(3) < 0
                    neg_count = neg_count + 1;
                    if display_print
                        fprintf(['Warning: negative sill/range at iteration %d ' ...
                                 '(count = %d), p = [%.4f %.4f %.4f]\n'], ...
                                iter+1, neg_count, p_new(1), p_new(2), p_new(3));
                    end
        
                    % stop_on_negative == true → stop after 10 consecutive negatives
                    if stop_on_negative && neg_count >= 10
                        if display_print
                            fprintf(['*** Stopping Newton iterations: sill or range ' ...
                                     'negative for 10 consecutive iterations.\n']);
                        end
                        aborted_negative = true;
                        break;  % exit while loop
                    end
                else
                    % Reset negative counter if values are positive
                    neg_count = 0;
                end
        
                % Store parameters
                p_estimates = [p_estimates, p_new];
        
                % Convergence: maximum absolute difference
                fark = max(abs(p_old - p_new));
        
                % Increase iteration counter
                iter = iter + 1;
        
                % ----- logging -----
                if display_print
                    elapsed_time = toc;
                    h = floor(elapsed_time/3600);
                    m = floor(mod(elapsed_time,3600)/60);
                    s = mod(elapsed_time,60);
            
                    fprintf(['Iter %2d | Δ = %.4f | p = [%.4f %.4f %.4f] | ' ...
                             'method = %s | time %02d:%02d:%02d\n'], ...
                            iter, fark, p_new(1), p_new(2), p_new(3), est_alg, h, m, round(s));
            
                    if fark <= 1e-4
                        fprintf('Convergence check: Δ <= 1e-4 at iteration %d.\n', iter);
                    end
                end
            end
        
            % ---------- outputs ----------
            elapsed_time = toc;
            p_final      = p_new;
        
            % Convergence flag and message
            if aborted_negative
                converged = false;
                if display_print
                    fprintf('Stopped early due to repeated negative sill/range values.\n');
                    fprintf('Last parameters used as output: [nugget = %.4f, sill = %.4f, alpha = %.4f]\n', ...
                             p_new(1), p_new(2), p_new(3));
                end
            else
                converged = (fark <= tol);
                if display_print
                    if converged
                        fprintf('Converged in %d iterations (Δ = %.4f <= tol = %.4g).\n', ...
                                iter, fark, tol);
                    else
                        fprintf('Stopped after reaching iteration limit (%d iterations). ', iter_limit);
                        fprintf('Convergence not achieved (Δ = %.4f > tol = %.4g).\n', fark, tol);
                    end
                end
            end
        
            % Final results display
            if display_print
                fprintf('nugget = %.4f\n', p_new(1));
                fprintf('sill   = %.4f\n', p_new(2));
                fprintf('alpha  = %.4f\n', p_new(3));
                fprintf('Method: %s, M-estimator: %s\n', est_alg, m_est_func);
                
                % Optional pretty time print, if a helper method exists in KRMLE_functions
                if exist('KRMLE_functions','class') && ismethod('KRMLE_functions','display_elapsed_time')
                    KRMLE_functions.display_elapsed_time(elapsed_time);
                end
            end
        end


        % ------------------------------------------------------------------
        % 2) LIKELIHOOD SCORING AND FISHER INFORMATION (CLASSICAL & ROBUST)
        % ------------------------------------------------------------------
        %
        %   This section provides:
        %
        %     (i)   classical score vector for the variogram parameters,
        %     (ii)  classical Fisher information matrix,
        %     (iii) robust score vector based on M-estimation,
        %     (iv)  robust Fisher information matrix.
        %
        %   All functions assume:
        %       - z       : n×1 vector of observations,
        %       - H_matrix: n×n distance matrix,
        %       - p       : parameter vector [nugget, sill, range],
        %       - v_type  : variogram model identifier
        %                  {'Lin','Mono','Gau','Exp','Sph','MSph','Cub'}.
        %
        %   The robust versions use:
        %       - standardized residuals r = V^{-1/2} z,
        %       - an M-estimator specified by m_est_func,
        %       - the function ROBUST_M_ESTIMATOR(r, m_est_func),
        %       - and the matrix square-root utility COMPUTE_V_SQRT.
        %
        %   These routines do not perform optimization by themselves; they
        %   provide gradients and curvature information for higher-level
        %   estimation algorithms (e.g., Newton–Raphson).
        %

        function scoring = calculate_scoring(z, H_matrix, p, v_type)
            % CALCULATE_SCORING
            % Computes the classical score function (gradient of the
            % log-likelihood) with respect to the variogram parameters.
            %
            % Inputs:
            %   z        : observation vector (n×1)
            %   H_matrix : distance matrix (n×n)
            %   p        : parameter vector [nugget, sill, range]
            %   v_type   : variogram type ('Lin','Mono','Gau','Exp','Sph','MSph','Cub')
            %
            % Output:
            %   scoring  : score vector (3×1)

            % Ensure z is a column vector
            z = z(:);

            % Covariance matrix and its derivatives
            [~, covariance_matrix, derivatives] = ...
                KRMLE_functions.calculate_variogram_variance_and_derivatives(H_matrix, p, v_type);

            % Extract covariance matrix and derivatives
            V   = covariance_matrix;
            V_p1 = derivatives.dC_dp1;  % ∂C/∂nugget
            V_p2 = derivatives.dC_dp2;  % ∂C/∂sill
            V_p3 = derivatives.dC_dp3;  % ∂C/∂range

            % Small regularization for numerical stability
            V = V + 1e-6 * eye(size(V));

            % Inverse covariance (more stable than inv(V))
            V_inv = V \ eye(size(V));

            % Initialize score vector
            scoring = zeros(3, 1);

            % Score for each parameter:
            %   S_i = -0.5 * trace(V^{-1} V_i) + 0.5 * z' V^{-1} V_i V^{-1} z
            V_derivs = {V_p1, V_p2, V_p3};

            for i = 1:3
                V_pi = V_derivs{i};
                scoring(i) = -0.5 * trace(V_inv * V_pi) + ...
                             0.5 * (z' * V_inv * V_pi * V_inv * z);
            end
        end

        function fisher = calculate_fisher(z, H_matrix, p, v_type)
            % CALCULATE_FISHER
            % Computes the (expected) Fisher information matrix for the
            % variogram parameters under the Gaussian likelihood.
            %
            % Inputs:
            %   z        : observation vector (n×1)  (not used in the
            %              expectation form, but kept for a consistent API)
            %   H_matrix : distance matrix (n×n)
            %   p        : parameter vector [nugget, sill, range]
            %   v_type   : variogram type ('Lin','Mono','Gau','Exp','Sph','MSph','Cub')
            %
            % Output:
            %   fisher   : Fisher information matrix (3×3)

            % Ensure z is a column vector (for a consistent interface)
            z = z(:); %#ok<NASGU>

            % Covariance matrix and its derivatives
            [~, covariance_matrix, derivatives] = ...
                KRMLE_functions.calculate_variogram_variance_and_derivatives(H_matrix, p, v_type);

            % Extract covariance matrix and derivatives
            V   = covariance_matrix;
            V_p1 = derivatives.dC_dp1;  % ∂C/∂nugget
            V_p2 = derivatives.dC_dp2;  % ∂C/∂sill
            V_p3 = derivatives.dC_dp3;  % ∂C/∂range

            % Small regularization for numerical stability
            V = V + 1e-6 * eye(size(V));

            % Inverse covariance
            V_inv = V \ eye(size(V));

            % Initialize Fisher information matrix
            fisher = zeros(3, 3);

            % Fisher(i,j) = 0.5 * trace(V^{-1} V_i V^{-1} V_j)
            V_derivs = {V_p1, V_p2, V_p3};

            for i = 1:3
                for j = 1:3
                    V_pi = V_derivs{i};
                    V_pj = V_derivs{j};
                    fisher(i, j) = 0.5 * trace(V_inv * V_pi * V_inv * V_pj);
                end
            end

            % Enforce symmetry (numerical precision)
            fisher = 0.5 * (fisher + fisher');

            % Sign convention as in the original code
            fisher = -fisher;
        end

        function robust_scoring = calculate_robust_scoring(z, H_matrix, p, m_est_func, v_type)
            % CALCULATE_ROBUST_SCORING
            % Computes a robust score vector using M-estimation, by replacing
            % quadratic residuals with a robust psi-function.
            %
            % Inputs:
            %   z          : observation vector (n×1)
            %   H_matrix   : distance matrix (n×n)
            %   p          : parameter vector [nugget, sill, range]
            %   m_est_func : robust M-estimator identifier
            %                (e.g. 'Huber','Tukey','Cauchy','Andrews', ...)
            %   v_type     : variogram type ('Lin','Mono','Gau','Exp','Sph','MSph','Cub')
            %
            % Output:
            %   robust_scoring : robust score vector (3×1)

            % Ensure z is a column vector
            z = z(:);

            % Covariance matrix and its derivatives
            [~, covariance_matrix, derivatives] = ...
                KRMLE_functions.calculate_variogram_variance_and_derivatives(H_matrix, p, v_type);

            % Extract covariance matrix and derivatives
            V   = covariance_matrix;
            V_p1 = derivatives.dC_dp1;  % ∂C/∂nugget
            V_p2 = derivatives.dC_dp2;  % ∂C/∂sill
            V_p3 = derivatives.dC_dp3;  % ∂C/∂range

            % Small regularization for numerical stability
            V = V + 1e-6 * eye(size(V));

            % Inverse covariance
            V_inv = V \ eye(size(V));

            % --- Compute matrix square root and standardized residuals ---
            [V_sqrt, V_sqrt_inv] = KRMLE_functions.compute_V_sqrt(V);
            r = V_sqrt_inv * z;

            % Robust M-estimator: rho(r), phi(r) = psi(r), weights w(r), K
            [rho, phi, weights, K] = KRMLE_functions.robust_m_estimator(r, m_est_func); %#ok<ASGLU>
            %#ok<NASGU>
            phi = phi(:);

            % Initialize robust score vector
            robust_scoring = zeros(3, 1);

            % Robust score for each parameter:
            %   S_i^R = -0.5 * trace(V^{-1} V_i) * K
            %           + 0.5 * phi' V^{1/2} V^{-1} V_i V^{-1} V^{1/2} phi
            V_derivs = {V_p1, V_p2, V_p3};

            for i = 1:3
                V_pi = V_derivs{i};
                robust_scoring(i) = -0.5 * trace(V_inv * V_pi) * K + ...
                                    0.5 * (phi' * V_sqrt * V_inv * V_pi * V_inv * V_sqrt * phi);
            end
        end

        function robust_fisher = calculate_robust_fisher(z, H_matrix, p, m_est_func, v_type)
            % CALCULATE_ROBUST_FISHER
            % Computes a robust analogue of the Fisher information matrix
            % under M-estimation, based on the chosen psi-function.
            %
            % Inputs:
            %   z          : observation vector (n×1)
            %   H_matrix   : distance matrix (n×n)
            %   p          : parameter vector [nugget, sill, range]
            %   m_est_func : robust M-estimator identifier
            %                (e.g. 'Huber','Tukey','Cauchy','Andrews', ...)
            %   v_type     : variogram type ('Lin','Mono','Gau','Exp','Sph','MSph','Cub')
            %
            % Output:
            %   robust_fisher : robust Fisher information matrix (3×3)

            % Ensure z is a column vector
            z = z(:);

            % Covariance matrix and its derivatives
            [~, covariance_matrix, derivatives] = ...
                KRMLE_functions.calculate_variogram_variance_and_derivatives(H_matrix, p, v_type);

            % Extract covariance matrix and derivatives
            V   = covariance_matrix;
            V_p1 = derivatives.dC_dp1;  % ∂C/∂nugget
            V_p2 = derivatives.dC_dp2;  % ∂C/∂sill
            V_p3 = derivatives.dC_dp3;  % ∂C/∂range

            % Small regularization for numerical stability
            V = V + 1e-6 * eye(size(V));

            % Inverse covariance
            V_inv = V \ eye(size(V));

            % --- Compute matrix square root and standardized residuals ---
            [V_sqrt, V_sqrt_inv] = KRMLE_functions.compute_V_sqrt(V);
            r = V_sqrt_inv * z;

            % Robust M-estimator: rho(r), phi(r), weights, K
            [rho, phi, weights, K] = KRMLE_functions.robust_m_estimator(r, m_est_func); %#ok<ASGLU,NASGU>

            % Initialize robust Fisher information matrix
            robust_fisher = zeros(3, 3);

            % Robust Fisher(i,j) ≈ 0.5 * trace(V^{-1} V_j V^{-1} V_i) * K
            V_derivs = {V_p1, V_p2, V_p3};

            for i = 1:3
                for j = 1:3
                    V_pi = V_derivs{i};
                    V_pj = V_derivs{j};
                    robust_fisher(i, j) = 0.5 * trace(V_inv * V_pj * V_inv * V_pi) * K;
                end
            end

            % Enforce symmetry (numerical precision)
            robust_fisher = 0.5 * (robust_fisher + robust_fisher');

            % Sign convention as in the original robust Fisher implementation
            robust_fisher = -robust_fisher;
        end


        % ------------------------------------------------------------------
        % 3) VARIOGRAM, COVARIANCE, AND PARAMETER DERIVATIVES
        % ------------------------------------------------------------------
        %
        %   This section implements several stationary isotropic variogram
        %   models and their corresponding covariance kernels, together with
        %   analytical first-order derivatives with respect to the model
        %   parameters.
        %
        %   Given:
        %       - a distance object H_matrix (scalar, vector, or n×n matrix),
        %       - a parameter vector p = [nugget, sill, range/alpha],
        %       - and a model type v_type ∈ {'Lin','Mono','Gau','Exp',
        %                                    'Sph','MSph','Cub'},
        %   the function CALCULATE_VARIOGRAM_VARIANCE_AND_DERIVATIVES returns:
        %
        %       - variogram_matrix  : γ(h), evaluated element-wise on H_matrix,
        %       - covariance_matrix : C(h), consistent with the chosen model,
        %       - derivatives       : struct with fields dC_dp1, dC_dp2, dC_dp3,
        %                             i.e. ∂C/∂nugget, ∂C/∂sill, ∂C/∂range.
        %
        %   The implementation:
        %       - supports scalar, vector, and (symmetric or non-symmetric)
        %         matrix inputs for H_matrix,
        %       - includes a generalized diagonal correction to enforce
        %         C(0) = nugget + sill on the covariance matrix,
        %       - is designed to be used in gradient-based likelihood or
        %         objective-function optimization for variogram fitting.
        %
        %   NOTE: This is a low-level building block; it does not perform any
        %   optimization by itself, but provides closed-form model evaluations
        %   and gradients required by higher-level estimation routines.
        %

        function [variogram_matrix, covariance_matrix, derivatives] = ...
            calculate_variogram_variance_and_derivatives(H_matrix, p, v_type)
        % Calculates variogram γ(h), covariance C(h), and first-order derivatives.
        %
        % Supports scalar, vector, or matrix inputs (symmetric or not).
        %
        % Inputs:
        %   H_matrix : distance matrix (n×n), vector (n×1), or scalar
        %   p        : [nugget, sill, range or alpha]
        %   v_type   : 'Lin','Mono','Gau','Exp','Sph','MSph','Cub'
        %
        % Outputs:
        %   variogram_matrix   : γ(h)
        %   covariance_matrix  : C(h)
        %   derivatives        : struct(dC_dp1, dC_dp2, dC_dp3)
        
            % --- Parameters ---
            nugget = p(1);
            sill   = p(2);
            if numel(p) >= 3
                alpha = p(3);
            else
                alpha = NaN;
            end
        
            h = H_matrix;
            sz = size(H_matrix);
        
            % --- Initialize outputs ---
            variogram_matrix  = zeros(sz);
            covariance_matrix = zeros(sz);
            derivatives = struct('dC_dp1', zeros(sz), ...
                                 'dC_dp2', zeros(sz), ...
                                 'dC_dp3', zeros(sz));
        
            % ============================================================
            % === VARIOGRAM MODELS =======================================
            % ============================================================
        
            switch v_type
                %% Linear
                case 'Lin'
                    variogram_matrix  = nugget + sill * h;
                    covariance_matrix = sill * (1 - h);
                    derivatives.dC_dp2 = (1 - h);
        
                %% Monomial
                case 'Mono'
                    variogram_matrix  = nugget + sill * (h.^alpha);
                    covariance_matrix = sill * (1-h.^alpha);
                    derivatives.dC_dp2 = (1-h.^alpha);
                    derivatives.dC_dp3 = -sill * (h.^alpha .* log(h + eps));  % eps avoids log(0) = -∞
        
                %% Gaussian
                case 'Gau'
                    arg = (h / alpha).^2;
                    exp_term = exp(-arg);
                    variogram_matrix  = nugget + sill * (1 - exp_term);
                    covariance_matrix = sill * exp_term;
                    derivatives.dC_dp2 = exp_term;
                    derivatives.dC_dp3 = 2 * sill * (h.^2) / (alpha^3) .* exp_term;
        
                %% Exponential
                case 'Exp'
                    arg = h / alpha;
                    exp_term = exp(-arg);
                    variogram_matrix  = nugget + sill * (1 - exp_term);
                    covariance_matrix = sill * exp_term;
                    derivatives.dC_dp2 = exp_term;
                    derivatives.dC_dp3 = sill * (h / (alpha^2)) .* exp_term;
        
                %% Spherical
                case 'Sph'
                    hr = h / alpha;
                    gamma_part = (1.5 * hr - 0.5 * hr.^3);
                    variogram_matrix  = nugget + sill * ((h < alpha) .* gamma_part + (h >= alpha));
                    covariance_matrix = sill * ((h < alpha) .* (1 - gamma_part));
                    derivatives.dC_dp2 = (h < alpha) .* (1 - gamma_part);
                    derivatives.dC_dp3 = sill * ((h < alpha) .* (1.5 * h / (alpha^2) - 1.5 * h.^3 / (alpha^4)));
        
                %% Modified Spherical
                case 'MSph'
                    hr = h / alpha;
                    gamma_part = (1.875 * hr - 1.25 * hr.^3 + 0.375 * hr.^5);
                    variogram_matrix  = nugget + sill * ((h < alpha) .* gamma_part + (h >= alpha));
                    covariance_matrix = sill * ((h < alpha) .* (1 - gamma_part));
                    derivatives.dC_dp2 = (h < alpha) .* (1 - gamma_part);
                    derivatives.dC_dp3 = sill * ((h < alpha) .* (1.875 * h / (alpha^2) ...
                        - 3.75 * h.^3 / (alpha^4) + 1.875 * h.^5 / (alpha^6)));
        
                %% Cubic
                case 'Cub'
                    hr = h / alpha;
                    gamma_part = (7 * hr.^2 - 8.75 * hr.^3 + 3.5 * hr.^5 - 0.75 * hr.^7);
                    variogram_matrix  = nugget + sill * ((h < alpha) .* gamma_part + (h >= alpha));
                    covariance_matrix = sill * ((h < alpha) .* (1 - gamma_part));
                    derivatives.dC_dp2 = (h < alpha) .* (1 - gamma_part);
                    derivatives.dC_dp3 = sill * ((h < alpha) .* (14 * h.^2 / (alpha^3) ...
                        - 26.25 * h.^3 / (alpha^4) + 17.5 * h.^5 / (alpha^6) - 5.25 * h.^7 / (alpha^8)));
        
                otherwise
                    error('Unknown variogram type: %s', v_type);
            end
        
            % ============================================================
            % === DIAGONAL / H = 0 CORRECTION (GENERALIZED) ==============
            % ============================================================
            if ismatrix(h) && sz(1) == sz(2) && isequal(h, h.') && numel(h) > 1
                % --- Symmetric square matrix (e.g., kriging distance matrix) ---
                diag_idx = 1:sz(1)+1:numel(h);
                covariance_matrix(diag_idx) = nugget + sill;
                derivatives.dC_dp1(diag_idx) = 1;
                derivatives.dC_dp2(diag_idx) = 1;
                derivatives.dC_dp3(diag_idx) = 0;
            else
                % --- Vector, scalar, or non-symmetric matrix ---
                zero_idx = (H_matrix == 0);
                if any(zero_idx(:))
                    covariance_matrix(zero_idx) = nugget + sill;
                    derivatives.dC_dp1(zero_idx) = 1;
                    derivatives.dC_dp2(zero_idx) = 1;
                    derivatives.dC_dp3(zero_idx) = 0;
                end
            end
        end


        % ------------------------------------------------------------------
        % 4) MATRIX SQUARE ROOT AND WHITENING OPERATORS
        % ------------------------------------------------------------------
        %
        %   This section provides a general-purpose utility to compute
        %   matrix square roots V^{1/2} and inverse square roots V^{-1/2}
        %   for symmetric covariance-like matrices.
        %
        %   The function COMPUTE_V_SQRT is mainly used to:
        %       - decorrelate / whiten residuals via t = V^{-1/2} u,
        %       - construct transformation matrices for influence-function
        %         based diagnostics and robust kriging procedures.
        %
        %   Multiple numerical back-ends are supported through the 'method'
        %   argument (eigen, svd, chol, sqrtm, ldl, newton, schur, polar,
        %   denman, cr). The function always:
        %       - symmetrizes and weakly regularizes V,
        %       - enforces real-valued outputs if small complex parts appear,
        %       - checks the relative Frobenius norm
        %             ||V^{1/2} V^{1/2} - V||_F / ||V||_F
        %         and issues a warning if the approximation is inaccurate.
        %
        %   This is a low-level numerical utility; it does NOT perform any
        %   variogram estimation or kriging by itself, but is designed to be
        %   reused across different robust estimation routines.
        %

        function [V_sqrt, V_sqrt_inv] = compute_V_sqrt(V, method)
            %COMPUTE_V_SQRT Computes matrix square root V^{1/2} and its inverse V^{-1/2}
            %               using various decomposition methods with enhanced numerical stability.
            %
            % Inputs:
            %   V       : covariance or distance matrix (n×n, symmetric)
            %   method  : string defining the decomposition method:
            %              'eigen'      → eigenvalue decomposition (default, safest)
            %              'svd'        → singular value decomposition (most robust)
            %              'chol'       → Cholesky decomposition (fastest for PD matrices)
            %              'sqrtm'      → MATLAB built-in matrix square root
            %              'ldl'        → LDL decomposition (for indefinite matrices)
            %              'newton'     → iterative Newton–Schulz method
            %              'schur'      → Schur decomposition (good for complex matrices)
            %              'polar'      → polar decomposition (SVD-based, very stable)
            %              'denman'     → Denman–Beavers iteration
            %              'cr'         → CR (Cayley) method
            %
            % Outputs:
            %   V_sqrt     : symmetric matrix square root of V (V_sqrt * V_sqrt ≈ V)
            %   V_sqrt_inv : inverse of matrix square root (V_sqrt_inv ≈ V^{-1/2})
        
            % --- Defaults and input validation ---
            if nargin < 2
                method = 'eigen';
            end
        
            n = size(V, 1);
            
            % === ALWAYS APPLY SYMMETRIZATION AND REGULARIZATION ===
            V = (V + V') / 2;                           % Always symmetrize
            V = V + eye(n) * (1e-10 * norm(V, 'fro'));  % Always regularize
            
            % Final symmetrization after regularization
            V = (V + V') / 2;
        
            % === Matrix square root computation methods ===
            switch lower(method)
                
                case 'eigen'
                    % Eigenvalue decomposition (recommended for general use)
                    [Q, D] = eig(V);
                    D = real(D);
                    D(D < 0) = 0;  % Clip negative eigenvalues
                    sqrt_D = sqrt(D);
                    inv_sqrt_D = diag(1 ./ (diag(sqrt_D) + eps));
                    
                    V_sqrt = Q * sqrt_D * Q';
                    V_sqrt_inv = Q * inv_sqrt_D * Q';
        
                case 'svd'
                    % Singular Value Decomposition (most numerically stable)
                    [U, S, ~] = svd(V);
                    S = real(S);
                    sqrt_S = sqrt(S);
                    inv_sqrt_S = diag(1 ./ (diag(sqrt_S) + eps));
                    
                    V_sqrt = U * sqrt_S * U';
                    V_sqrt_inv = U * inv_sqrt_S * U';
        
                case 'chol'
                    % Cholesky decomposition (fastest for positive definite matrices)
                    try
                        L = chol(V, 'lower');
                        V_sqrt = L;
                        V_sqrt_inv = inv(L);
                    catch
                        warning('COMPUTE_V_SQRT:CholeskyFailed', ...
                               'Cholesky decomposition failed. Falling back to eigen method.');
                        [V_sqrt, V_sqrt_inv] = KRMLE_functions.compute_V_sqrt(V, 'eigen');
                        return;
                    end
        
                case 'sqrtm'
                    % MATLAB built-in matrix square root
                    V_sqrt = sqrtm(V);
                    V_sqrt_inv = inv(V_sqrt);
        
                case 'ldl'
                    % LDL decomposition (handles indefinite matrices)
                    [L, D] = ldl(V);
                    D(D < 0) = 0;  % Clip negative eigenvalues
                    sqrt_D = sqrt(D);
                    inv_sqrt_D = diag(1 ./ (diag(sqrt_D) + eps));
                    
                    V_sqrt = L * sqrt_D * L';
                    V_sqrt_inv = L * inv_sqrt_D * L';
        
                case 'newton'
                    % Newton–Schulz iterative method
                    n = size(V,1);
                    I = eye(n);
                    X = I;  % Initial guess for V^{-1/2}
                    for k = 1:10
                        X = 0.5 * X * (3*I - V * X * X);
                    end
                    V_sqrt_inv = X;
                    V_sqrt = inv(X);
        
                case 'schur'
                    % Schur decomposition (good for complex matrices)
                    [U, T] = schur(V);
                    T_sqrt = diag(sqrt(diag(T)));
                    V_sqrt = U * T_sqrt * U';
                    V_sqrt_inv = U * diag(1./diag(T_sqrt)) * U';
        
                case 'polar'
                    % Polar decomposition (SVD-based, very stable)
                    [U, S, V_mat] = svd(V);
                    sqrt_S = diag(sqrt(diag(S)));
                    V_sqrt = U * sqrt_S * V_mat';
                    V_sqrt_inv = V_mat * diag(1./diag(sqrt_S)) * U';
        
                case 'denman'
                    % Denman–Beavers iteration (simultaneous sqrt and inv_sqrt)
                    n = size(V,1);
                    Y = V;
                    Z = eye(n);
                    for k = 1:20
                        Y_new = 0.5 * (Y + inv(Z));
                        Z_new = 0.5 * (Z + inv(Y));
                        Y = Y_new;
                        Z = Z_new;
                    end
                    V_sqrt = Y;
                    V_sqrt_inv = Z;
        
                case 'cr'
                    % CR (Cayley) method
                    n = size(V,1);
                    X = V;
                    Y = eye(n);
                    for k = 1:15
                        X_new = 0.5 * X * (3*eye(n) - Y * X);
                        Y_new = 0.5 * (3*eye(n) - Y * X) * Y;
                        X = X_new;
                        Y = Y_new;
                    end
                    V_sqrt = X;
                    V_sqrt_inv = Y;
        
                otherwise
                    error('COMPUTE_V_SQRT:UnknownMethod', ...
                          'Unknown method: %s. Choose: eigen, svd, chol, sqrtm, ldl, newton, schur, polar, denman, or cr.', method);
            end
        
            % === Enhanced output validation ===
            
            % Check for complex parts and force real output if needed
            if any(imag(V_sqrt(:)) ~= 0) || any(imag(V_sqrt_inv(:)) ~= 0)
                warning('COMPUTE_V_SQRT:ComplexOutput', ...
                       'Matrix square root contains complex parts. Forcing real output.');
                
                V_sqrt = real(V_sqrt);
                V_sqrt_inv = real(V_sqrt_inv);
            end
        
            % Verify numerical accuracy
            residual_norm = norm(V_sqrt * V_sqrt - V, 'fro') / norm(V, 'fro');
            if residual_norm > 1e-6
                warning('COMPUTE_V_SQRT:AccuracyWarning', ...
                       'Matrix square root accuracy may be low. Residual norm: %.2e', residual_norm);
            end
        
            % Final symmetry enforcement for output
            V_sqrt = (V_sqrt + V_sqrt') / 2;
            V_sqrt_inv = (V_sqrt_inv + V_sqrt_inv') / 2;
        end


        % ------------------------------------------------------------------
        % 5) ROBUST M-ESTIMATION LOSS AND WEIGHT FUNCTIONS
        % ------------------------------------------------------------------
        %
        %   This section provides low-level building blocks for robust
        %   estimation. The function ROBUST_M_ESTIMATOR computes:
        %
        %       - rho(r)     : loss function values
        %       - phi(r)     : influence function psi(r) = d rho(r) / dr
        %       - weights(r) : IRLS weights w(r) = psi(r) / r
        %       - K          : E[psi(Z)^2] for Z ~ N(0,1), used in scale
        %                     equations to ensure consistency under normality.
        %
        %   It supports several classical and redescending M-estimators
        %   (Huber, Tukey, Cauchy, , insha, Logistic, etc.) via the
        %   'm_est_func' string identifier. This is a generic utility and
        %   does NOT perform any variogram fitting or kriging by itself.
        %

        function [rho, phi, weights, K] = robust_m_estimator(r, m_est_func)
            % ROBUST_M_ESTIMATOR
            %   Compute rho(r), phi(r) = psi(r), IRLS weights w(r) = psi(r)/r,
            %   and the normalizing constant K = E[psi(Z)^2], Z ~ N(0,1),
            %   for a given M-estimation loss function.
            %
            %   INPUTS
            %     r          : residuals (scalar or vector, will be reshaped to column)
            %     m_est_func : name of the M-estimator (e.g. 'OLS', 'insha', 'Cauchy', ...)
            %
            %   OUTPUTS
            %     rho     : loss values rho(r)
            %     phi     : influence function psi(r) = d rho(r) / dr
            %     weights : IRLS weights w(r) = psi(r) / r (with stable handling near r = 0)
            %     K       : theoretical value of E[psi(Z)^2] for Z ~ N(0,1),
            %               used in the M-scale equation for consistency.
        
            r = r(:);  % ensure r is a column vector
            n = length(r);
        
            % Small threshold to avoid division by values extremely close to zero.
            % This improves numerical stability in weights near r = 0.
            eps_val = 1e-8;
        
            % Initialize outputs
            rho     = zeros(n, 1);
            phi     = zeros(n, 1);
            weights = zeros(n, 1);
            K       = 0;
        
            % Select M-estimator
            if strcmp(m_est_func, 'OLS')
                rho     = r.^2 / 2;
                phi     = r;
                weights = ones(n, 1);
                integrand = @(x) (x.^2) .* (2*pi)^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -Inf, Inf);
        
            elseif strcmp(m_est_func, 'LAD')
                rho = abs(r);
                phi = sign(r);
                % Basic weight definition: w = psi(r) / r = sign(r) / r = 1/|r|
                weights = 1 ./ abs(r);
                % For |r| very close to zero, set weight = 1 to avoid blow-up.
                small_idx = abs(r) < eps_val;
                weights(small_idx) = 1;
        
                integrand = @(x) (sign(x)).^2 .* (2*pi)^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -Inf, Inf);
        
            elseif strcmp(m_est_func, 'L1L2')
                rho = 2 * (sqrt(1 + (r.^2 / 2)) - 1);
                phi = r ./ sqrt(1 + (r.^2 / 2));
                weights = 1 ./ sqrt(1 + r.^2/2);
                integrand = @(x) (x ./ sqrt(1 + (x.^2 / 2))).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -Inf, Inf);
        
            elseif strcmp(m_est_func, 'Fair')
                c = 1.3998;
                rho = c^2 * (abs(r) / c - log(1 + (abs(r) / c)));
                phi = r ./ (1 + abs(r) / c);
                weights = 1 ./ (1 + abs(r)/c);
                integrand = @(x) (x ./ (1 + abs(x)/c)).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -Inf, Inf);
        
            elseif strcmp(m_est_func, 'Cauchy')
                c = 2.3849;
                rho = c^2 / 2 * log(1 + (r / c).^2);
                phi = r ./ (1 + (r / c).^2);
                weights = 1 ./ (1 + (r/c).^2);
                integrand = @(x) (x ./ (1 + (x ./ c).^2)).^2 .* (2*pi)^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -Inf, Inf);
        
            elseif strcmp(m_est_func, 'Geman_McClure')
                rho = (r.^2 / 2) ./ (1 + r.^2);
                phi = r ./ (1 + r.^2).^2;
                weights = 1 ./ (1 + r.^2).^2;
                integrand = @(x) (x ./ (1 + x.^2).^2).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -Inf, Inf);
        
            elseif strcmp(m_est_func, 'Welsch')
                c = 2.9846;
                rho = c^2 / 2 * (1 - exp(-(r / c).^2));
                phi = r .* exp(-(r / c).^2);
                weights = exp(-(r / c).^2);
                integrand = @(x) (x .* exp(-(x ./ c).^2)).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -Inf, Inf);
        
            elseif strcmp(m_est_func, 'Huber')
                c = 1.345;
                idx = abs(r) <= c;
                rho(idx)  = r(idx).^2 / 2;
                rho(~idx) = c * abs(r(~idx)) - c^2 / 2;
        
                phi(idx)  = r(idx);
                phi(~idx) = c * sign(r(~idx));
        
                weights(idx)  = 1;
                weights(~idx) = c ./ abs(r(~idx));
        
                integrand_1 = @(x) (x.^2) .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                integrand_2 = @(x) (c * sign(x)).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand_1, -c, c) + integral(integrand_2, -Inf, -c) + integral(integrand_2, c, Inf);
        
            elseif strcmp(m_est_func, 'Tukey')
                c = 4.6851;
                idx = abs(r) <= c;
                rho(idx)  = (c^2 / 6) * (1 - (1 - (r(idx) / c).^2).^3);
                rho(~idx) = c^2 / 6;
        
                phi(idx)  = r(idx) .* (1 - (r(idx) / c).^2).^2;
                phi(~idx) = 0;
        
                weights(idx)  = (1 - (r(idx) / c).^2).^2;
                weights(~idx) = 0;
        
                integrand = @(x) ((x .* (1 - (x / c).^2).^2).^2) .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -c, c);
        
            elseif strcmp(m_est_func, 'Bell')
                rho = (5 * 4) * (1 - (1 + (r.^2 / 5)).^(-2));
                phi = r .* (1 + (r.^2 / 5)).^(-3);
                weights = 1 ./ (1 + r.^2/5).^3;
                integrand = @(x) ((x .* (1 + (x.^2 / 5)).^(-3)).^2) .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -Inf, Inf);
        
            elseif strcmp(m_est_func, 'Bell2')
                c = 2.1522;
                rho = (5 * c^2 / 4) * (1 - (1 + (r.^2 / (5 * c^2))).^(-2));
                phi = r .* (1 + (r.^2 / (5*c^2))).^(-3);
                weights = 1 ./ (1 + r.^2/(5*c^2)).^3;
                integrand = @(x) (x .* (1 + (x.^2 / (5*c^2))).^(-3)).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -Inf, Inf);
        
            elseif strcmp(m_est_func, 'Logistic')
                c = 1.205;
                rho = c^2 * log(cosh(r/c));
                phi = c * tanh(r / c);
                % Basic weight definition: w = psi(r) / r = (c * tanh(r/c)) / r
                weights = (c * tanh(r / c)) ./ r;
                % For |r| very close to zero, set weight = 1 (continuous limit of psi(r)/r).
                small_idx = abs(r) < eps_val;
                weights(small_idx) = 1;
        
                integrand = @(x) (c * tanh(x ./ c)).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -Inf, Inf);
        
            elseif strcmp(m_est_func, 'insha')
                c = 4.685;
                rho = (c^2 / 4) * (atan(r/c).^2 + (c^2 * r.^2) ./ (c^4 + r.^4));
                phi = r .* (1 + (r / c).^4).^(-2);
                weights = 1 ./ (1 + (r / c).^4).^2;
                integrand = @(x) (x .* (1 + (x.^4 / c^4)).^(-2)).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -Inf, Inf);
        
            elseif strcmp(m_est_func, 'Andrews')
                c = 1.339;
                idx = abs(r) <= c*pi;
        
                rho(idx)  = c^2 * (1 - cos(r(idx)/c));
                rho(~idx) = 2*c^2;
        
                phi(idx)  = c * sin(r(idx) / c);
                phi(~idx) = 0;
        
                % Basic weight definition: w = psi(r) / r = (c * sin(r/c)) / r
                % Use r(idx)/c in the denominator to keep vector sizes consistent.
                weights(idx)  = sin(r(idx) / c) ./ (r(idx) / c);
                weights(~idx) = 0;
        
                % For |r| very close to zero, set weight = 1 (limit of psi(r)/r as r -> 0).
                small_idx = abs(r) < eps_val;
                weights(small_idx) = 1;
        
                integrand = @(x) (c * sin(x/c)).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -c*pi, c*pi);
        
            elseif strcmp(m_est_func, 'Hampel')
                a = 2.0; b = 4.0; c = 8.0;
                idx1 = abs(r) <= a;
                idx2 = (abs(r) > a) & (abs(r) <= b);
                idx3 = (abs(r) > b) & (abs(r) <= c);
                idx4 = abs(r) > c;
        
                rho(idx1) = r(idx1).^2 / 2;
                rho(idx2) = a * abs(r(idx2)) - a^2 / 2;
                rho(idx3) = (a * (b^2 - abs(r(idx3)).^2)) / (2*(b-a)) + a*(a+b)/2;
                rho(idx4) = a * (b + c) / 2;
        
                phi(idx1) = r(idx1);
                phi(idx2) = a * sign(r(idx2));
                phi(idx3) = a * (b - abs(r(idx3))) .* sign(r(idx3)) / (b - a);
                phi(idx4) = 0;
        
                weights(idx1) = 1;
                weights(idx2) = a ./ abs(r(idx2));
                weights(idx3) = a * (b - abs(r(idx3))) ./ (abs(r(idx3)) * (b - a));
                weights(idx4) = 0;
        
                integrand1 = @(x) (x.^2) .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                integrand2 = @(x) (a * sign(x)).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                integrand3 = @(x) (a * (b - abs(x)) .* sign(x) / (b - a)).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
        
                K = integral(integrand1, -a, a) + integral(integrand2, -b, -a) + integral(integrand2, a, b) + ...
                    integral(integrand3, -c, -b) + integral(integrand3, b, c);
        
            elseif strcmp(m_est_func, 'Bisquare')
                c = 4.685;
                idx = abs(r) <= c;
                rho(idx)  = (c^2/6) * (1 - (1 - (r(idx)/c).^2).^3);
                rho(~idx) = c^2/6;
        
                phi(idx)  = r(idx) .* (1 - (r(idx) / c).^2).^2;
                phi(~idx) = 0;
        
                weights(idx)  = (1 - (r(idx) / c).^2).^2;
                weights(~idx) = 0;
        
                integrand = @(x) (x .* (1 - (x/c).^2).^2).^2 .* (2*pi).^(-0.5) .* exp(-x.^2 / 2);
                K = integral(integrand, -c, c);
        
            else
                error('Unknown M-estimator function: %s', m_est_func);
            end
        end


        % ------------------------------------------------------------------
        % 6) UTILITY: PRETTY PRINT OF ELAPSED TIME
        % ------------------------------------------------------------------
        %
        %   Small helper to display the total elapsed time in a readable
        %   format (hours / minutes / seconds). This is used by
        %   ESTIMATE_VARIAGRAM_PARAMETERS when 'display_print' is true.
        %

        function display_elapsed_time(elapsed_time)
            % DISPLAY_ELAPSED_TIME
            % Convert elapsed_time (in seconds) to hours, minutes and seconds
            % and print a human-readable message to the command window.
        
            % Convert to hours, minutes, and seconds
            hours   = floor(elapsed_time / 3600);                 % whole hours
            minutes = floor(mod(elapsed_time, 3600) / 60);        % whole minutes
            seconds = mod(elapsed_time, 60);                      % remaining seconds
        
            % Check if elapsed time is greater than or equal to 1 hour
            if hours >= 1
                % Display the time including hours, minutes, and seconds
                disp(['Total elapsed time: ', ...
                      num2str(hours),   ' hours ', ...
                      num2str(minutes), ' minutes ', ...
                      num2str(seconds, '%.2f'), ' seconds']);
            else
                % Display the time in minutes and seconds only
                disp(['Total elapsed time: ', ...
                      num2str(minutes), ' minutes ', ...
                      num2str(seconds, '%.2f'), ' seconds']);
            end
        end


    end

end

