

% solve_subproblem1_S_W.m
function [s_k_opt, W_k_opt] = solve_subproblem1_S_W(q_current, s_k_initial, W_k_initial, params)
    N = params.N; K = params.K; V = params.V; M_ris = params.M;
    P_UAV_linear = params.P_UAV_linear;
    sigma_k_sq_linear = params.sigma_k_sq_linear;
    sigma_s_sq_linear = params.sigma_s_sq_linear;
    R_min = params.R_min; Gamma_SNR_linear = params.Gamma_SNR_linear;
    beta_C = params.beta_C; beta_S = 1 - beta_C;
    I_max_sca_sub1 = params.I_max_sca_sub1; epsilon_sca_conv = params.epsilon_sca;
    u_R_coords_const = params.u_R_pos;
    u_k_all_coords_const = params.u_k_coords_3d;
    u_s_coords_const = params.u_s_3d;
    s_k_iter = s_k_initial;
    W_k_iter = W_k_initial;
    obj_val_prev_sca = -Inf;

    fprintf('    Solving Subproblem 1 (S, W_k) using SCA...\n');
    Phi_all_val_ris = calculate_ris_phase_shifts(params, q_current);

    for sca_iter_idx = 1:I_max_sca_sub1
        fprintf('      SCA Iteration %d for Subproblem 1\n', sca_iter_idx);

        H_k_eff_all = zeros(V, V, K, N);
        H_s_eff_term_all = zeros(V, V, N);
        for n_pre = 1:N
            q_n_val = q_current(:, n_pre);
            Phi_n_val = diag(Phi_all_val_ris(n_pre, :));
            [H_AR_n, h_k_AC_all, h_AS_n, h_k_RC_all, h_RS_n] = calculate_channels(q_n_val, u_R_coords_const, u_k_all_coords_const, u_s_coords_const, params);
            h_s_eff = h_AS_n;
            if M_ris > 0, h_s_eff = h_s_eff + H_AR_n' * Phi_n_val * h_RS_n; end
            H_s_matrix_eff = h_s_eff * h_s_eff';
            H_s_eff_term_all(:,:,n_pre) = H_s_matrix_eff' * H_s_matrix_eff;
            for k_pre = 1:K
                h_k_eff = h_k_AC_all(:, k_pre);
                if M_ris > 0, h_k_eff = h_k_eff + H_AR_n' * Phi_n_val * h_k_RC_all(:, k_pre); end
                H_k_eff_all(:,:,k_pre,n_pre) = h_k_eff * h_k_eff';
            end
        end

        cvx_begin
            cvx_solver mosek;
            variable s_k_cvx(K, N)
            variable W_k_cvx(V, V, K, N) hermitian semidefinite
            expression R_k_lb_matrix(K, N)
            expression sensing_snr_total
            sensing_snr_total = 0;
            for n = 1:N
                H_s_n_term = squeeze(H_s_eff_term_all(:,:,n));
                W_total_n_cvx = sum(W_k_cvx(:,:,:,n), 3);
                sensing_snr_total = sensing_snr_total + real(trace(H_s_n_term * W_total_n_cvx));
                W_k_n_iter_all = squeeze(W_k_iter(:,:,:,n));
                s_k_n_iter = s_k_iter(:, n);
                for k = 1:K
                    H_k_n_eff = squeeze(H_k_eff_all(:,:,k,n));
                    W_k_n_cvx_user = squeeze(W_k_cvx(:,:,k,n));
                    W_k_n_iter_user = squeeze(W_k_n_iter_all(:,:,k));
                    signal_at_iter_k_n = real(trace(H_k_n_eff * W_k_n_iter_user));
                    rate_log_lb = log2(1 + signal_at_iter_k_n/sigma_k_sq_linear) + (1 / (log(2)*(sigma_k_sq_linear + signal_at_iter_k_n))) * real(trace(H_k_n_eff * (W_k_n_cvx_user - W_k_n_iter_user)));
                    R_k_lb_matrix(k, n) = s_k_n_iter(k) * rate_log_lb + log2(1 + signal_at_iter_k_n/sigma_k_sq_linear) * (s_k_cvx(k,n) - s_k_n_iter(k));
                end
            end
            sensing_objective = beta_S * (sensing_snr_total / sigma_s_sq_linear);
            comm_objective = beta_C * sum(R_k_lb_matrix(:));
            maximize(comm_objective + sensing_objective);
            subject to
                0 <= s_k_cvx <= 1;
                2 * s_k_iter .* s_k_cvx - s_k_iter.^2 - s_k_cvx <= 0;
                for n_c = 1:N
                    sum(s_k_cvx(:, n_c)) <= 1;
                    real(trace(sum(W_k_cvx(:,:,:,n_c), 3))) <= P_UAV_linear;
                end
                
                % (sensing_snr_total / sigma_s_sq_linear) / N >= Gamma_SNR_linear;
                % for k_c = 1:K
                %     sum(R_k_lb_matrix(k_c, :)) / N >= R_min;
                % end
                delta_W = 5; % This is now a per-matrix limit, so it can be smaller
                for n_tr = 1:N
                   for k_tr = 1:K
                    % Apply the norm constraint to each V x V slice
                    norm(squeeze(W_k_cvx(:,:,k_tr,n_tr)) - squeeze(W_k_iter(:,:,k_tr,n_tr)), 'fro') <= delta_W;
                   end
                end   



        cvx_end

        if contains(cvx_status, 'Solved')
            if abs(cvx_optval - obj_val_prev_sca) / (abs(obj_val_prev_sca) + 1e-9) < epsilon_sca_conv, s_k_iter = s_k_cvx; W_k_iter = W_k_cvx; break; end
            s_k_iter = s_k_cvx; W_k_iter = W_k_cvx; obj_val_prev_sca = cvx_optval;
        else
            fprintf('      Subproblem 1 CVX failed with status %s. Aborting.\n', cvx_status);
            s_k_opt = s_k_initial; W_k_opt = W_k_initial; return;
        end
    end
    s_k_opt = zeros(K, N);
    for n = 1:N
        [~, idx] = max(s_k_iter(:, n)); if s_k_iter(idx,n) > 0.5, s_k_opt(idx, n) = 1; end
    end
    W_k_opt = zeros(V, V, K, N);
    for n = 1:N
        for k = 1:K
            if s_k_opt(k, n) == 1
                W_k_val = squeeze(W_k_iter(:,:,k,n));
                if any(isnan(W_k_val),'all') || any(isinf(W_k_val),'all'), continue; end
                W_k_val = (W_k_val + W_k_val') / 2;
                [V_eig, D_eig] = eig(W_k_val);
                [lambda_max, idx_max] = max(real(diag(D_eig)));
                if lambda_max > 1e-9
                    w_max = sqrt(lambda_max) * V_eig(:, idx_max);
                    W_k_opt(:,:,k,n) = w_max * w_max';
                end
            end
        end
    end
end










































































% % solve_subproblem1_S_W.m
% 
% function [s_k_opt, W_k_opt] = solve_subproblem1_S_W(q_current, s_k_initial, W_k_initial, params)
%     % Unpack parameters
%     N = params.N; K = params.K; V = params.V; M_ris = params.M;
%     P_UAV_linear = params.P_UAV_linear;
%     sigma_k_sq_linear = params.sigma_k_sq_linear;
%     sigma_s_sq_linear = params.sigma_s_sq_linear;
%     R_min = params.R_min; 
%     Gamma_SNR_linear = params.Gamma_SNR_linear;
%     beta_C = params.beta_C; 
%     beta_S = 1 - beta_C;
%     I_max_sca_sub1 = params.I_max_sca_sub1; 
%     epsilon_sca_conv = params.epsilon_sca;
%     
%     % Get fixed coordinates
%     u_R_coords_const = params.u_R_pos;
%     u_k_all_coords_const = params.u_k_coords_3d;
%     u_s_coords_const = params.u_s_3d;
% 
%     % Initialize SCA iteration variables
%     s_k_iter = s_k_initial;
%     W_k_iter = W_k_initial;
%     obj_val_prev_sca = -Inf;
% 
%     fprintf('    Solving Subproblem 1 (S, W_k) using SCA...\n');
% 
%     % Precompute phase shifts for the fixed trajectory
%     Phi_all_val_ris = calculate_ris_phase_shifts(params, q_current);
% 
%     for sca_iter_idx = 1:I_max_sca_sub1
%         fprintf('      SCA Iteration %d for Subproblem 1\n', sca_iter_idx);
% 
%         % Pre-compute all effective channels BEFORE entering the CVX block
%         H_k_eff_all = zeros(V, V, K, N);
%         H_s_eff_term_all = zeros(V, V, N);
%         for n_pre = 1:N
%             q_n_val = q_current(:, n_pre);
%             Phi_n_val = diag(Phi_all_val_ris(n_pre, :));
%             [H_AR_n, h_k_AC_all, h_AS_n, h_k_RC_all, h_RS_n] = ...
%                 calculate_channels(q_n_val, u_R_coords_const, u_k_all_coords_const, u_s_coords_const, params);
%             
%             h_s_eff = h_AS_n;
%             if M_ris > 0, h_s_eff = h_s_eff + H_AR_n' * Phi_n_val * h_RS_n; end
%             H_s_matrix_eff = h_s_eff * h_s_eff';
%             H_s_eff_term_all(:,:,n_pre) = H_s_matrix_eff' * H_s_matrix_eff;
%             
%             for k_pre = 1:K
%                 h_k_eff = h_k_AC_all(:, k_pre);
%                 if M_ris > 0, h_k_eff = h_k_eff + H_AR_n' * Phi_n_val * h_k_RC_all(:, k_pre); end
%                 H_k_eff_all(:,:,k_pre,n_pre) = h_k_eff * h_k_eff';
%             end
%         end
%         cvx_clear
%         cvx_begin
%             cvx_solver mosek;
% 
%             % --- CVX Variables ---
%             variable s_k_cvx(K, N)
%             variable W_k_cvx(V, V, K, N) hermitian semidefinite
% 
%             % --- CVX Expressions ---
%             expression R_k_lb_matrix(K, N)
%             expression sensing_snr_total
% 
%             % --- Build Expressions ---
%             sensing_snr_total = 0;
% 
%             for n = 1:N
%                 H_s_n_term = squeeze(H_s_eff_term_all(:,:,n));
%                 W_total_n_cvx = sum(W_k_cvx(:,:,:,n), 3);
%                 sensing_snr_total = sensing_snr_total + real(trace(H_s_n_term * W_total_n_cvx));
%                 
%                 W_k_n_iter_all = squeeze(W_k_iter(:,:,:,n));
%                 s_k_n_iter = s_k_iter(:, n);
%                 for k = 1:K
%                     H_k_n_eff = squeeze(H_k_eff_all(:,:,k,n));
%                     W_k_n_cvx_user = squeeze(W_k_cvx(:,:,k,n));
%                     W_k_n_iter_user = squeeze(W_k_n_iter_all(:,:,k));
% 
%                     signal_at_iter_k_n = real(trace(H_k_n_eff * W_k_n_iter_user));
%                     
%                     rate_log_lb = log2(1 + signal_at_iter_k_n/sigma_k_sq_linear) + ...
%                                   (1 / (log(2)*(sigma_k_sq_linear + signal_at_iter_k_n))) * ...
%                                   real(trace(H_k_n_eff * (W_k_n_cvx_user - W_k_n_iter_user)));
%                     
%                     R_k_lb_matrix(k, n) = s_k_n_iter(k) * rate_log_lb + ...
%                                           log2(1 + signal_at_iter_k_n/sigma_k_sq_linear) * (s_k_cvx(k,n) - s_k_n_iter(k));
%                 end
%             end
% 
%             % --- Objective Function ---
%             sensing_objective = beta_S * (sensing_snr_total / sigma_s_sq_linear);
%             comm_objective = beta_C * sum(R_k_lb_matrix(:));
%             
%             maximize(comm_objective + sensing_objective);
% 
%             % --- Constraints ---
%             subject to
%                 0 <= s_k_cvx <= 1; % C1
% 
%                 % Linearized binary constraint
%                 2 * s_k_iter .* s_k_cvx - s_k_iter.^2 - s_k_cvx <= 0;
% 
%                 for n_c = 1:N % Per-slot constraints
%                     sum(s_k_cvx(:, n_c)) <= 1; % C2
%                     real(trace(sum(W_k_cvx(:,:,:,n_c), 3))) <= P_UAV_linear; % C3a
%                 end
% 
%                 % ... (All existing constraints C1, C2, C3a, etc. remain) ...
% 
%         % ================================================================
%         % === TRUST REGION CONSTRAINT (THE FIX) ===
%         % ================================================================
%         % Limit how much the new beamformers can change from the previous iteration.
%         % This prevents the solver from running to infinity.
%         % 'delta_W' is a tuning parameter, start with a reasonable value.
%         delta_W = 10; 
%         for n_c_tr = 1:N
%             for k_c_tr = 1:K
%                 % Use Frobenius norm for the difference between matrices
%                 norm(W_k_cvx(:,:,k_c_tr,n_c_tr) - W_k_iter(:,:,k_c_tr,n_c_tr), 'fro') <= delta_W;
%             end
%         end
%         % ================================================================
% 
%         cvx_end
% 
%         if contains(cvx_status, 'Solved')
%             if abs(cvx_optval - obj_val_prev_sca) / (abs(obj_val_prev_sca) + 1e-9) < epsilon_sca_conv
%                 s_k_iter = s_k_cvx; W_k_iter = W_k_cvx; break;
%             end
%             s_k_iter = s_k_cvx; W_k_iter = W_k_cvx; obj_val_prev_sca = cvx_optval;
%         else
%             fprintf('      Subproblem 1 CVX failed with status %s. Aborting.\n', cvx_status);
%             s_k_opt = s_k_initial; W_k_opt = W_k_initial; return;
%         end
%     end
%     
%     % Post-processing
%     s_k_opt = zeros(K, N);
%     for n = 1:N
%         [max_val, idx] = max(s_k_iter(:, n));
%         if max_val > 0.5, s_k_opt(idx, n) = 1; end
%     end
% 
%     W_k_opt = zeros(V, V, K, N);
%     for n = 1:N
%         for k = 1:K
%             if s_k_opt(k, n) == 1
%                 W_k_val = squeeze(W_k_iter(:,:,k,n));
%                 if any(isnan(W_k_val), 'all') || any(isinf(W_k_val), 'all'), continue; end
%                 W_k_val = (W_k_val + W_k_val') / 2;
%                 [V_eig, D_eig] = eig(W_k_val);
%                 [lambda_max, idx_max] = max(real(diag(D_eig)));
%                 if lambda_max > 1e-9
%                     w_max = sqrt(lambda_max) * V_eig(:, idx_max);
%                     W_k_opt(:,:,k,n) = w_max * w_max';
%                 end
%             end
%         end
%     end
% end