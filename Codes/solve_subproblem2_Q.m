% solve_subproblem2_Q.m
function q_opt = solve_subproblem2_Q(params, s_k_fixed, W_k_fixed, q_initial)
    N = params.N; K = params.K;
    I_max_sca_local = params.I_max_sca_sub2;
    epsilon_sca_conv = params.epsilon_sca;
    q_iter = q_initial;
    obj_val_prev_sca = -Inf;

    fprintf('    Solving Subproblem 2 (Q) using SCA...\n');

    for sca_iter_idx = 1:I_max_sca_local
        fprintf('      SCA Iteration %d for Subproblem 2\n', sca_iter_idx);
        
        coeffs = precompute_sca_coeffs_for_Q(params, q_iter, s_k_fixed, W_k_fixed);
        [u_iter, z_iter, t_k_iter] = calculate_slack_variables(params, q_iter);

        cvx_begin
            cvx_solver mosek;
            variable q_cvx(3, N)
            variable t_k_cvx(K, N) nonnegative
            variable u_cvx(1, N) nonnegative
            variable z_cvx(1, N) nonnegative
            expression comm_rate_lb_total;
            expression sensing_snr_lb_total;
            comm_rate_lb_total = 0;
            sensing_snr_lb_total = 0;
            for n = 1:N
                sensing_snr_lb_n = coeffs.SNRs_r2(n) + coeffs.A1_n(n) * (u_cvx(n) - u_iter(n)) + coeffs.A2_n(n) * (z_cvx(n) - z_iter(n));
                sensing_snr_lb_total = sensing_snr_lb_total + sensing_snr_lb_n;
                for k = 1:K
                    if s_k_fixed(k, n) == 1
                        comm_rate_lb_kn = coeffs.Rk_r2(k, n) + coeffs.R1_n(k, n) * (t_k_cvx(k, n) - t_k_iter(k, n)) + coeffs.R2_n(k, n) * (u_cvx(n) - u_iter(n));
                        comm_rate_lb_total = comm_rate_lb_total + comm_rate_lb_kn;
                    end
                end
            end
            maximize(params.beta_C * (comm_rate_lb_total/N) + (1-params.beta_C) * (sensing_snr_lb_total/N));
            subject to
                % (QoS constraints are disabled for debugging)
                % sensing_snr_lb_total / N >= params.Gamma_SNR_linear;
                % for k_c = 1:K
                %     ...
                % end

                for n_c = 1:N-1
                    norm(q_cvx(:, n_c+1) - q_cvx(:, n_c)) <= params.V_max * params.tau;
                end
                q_cvx(:, 1) == params.q0;
                q_cvx(:, N) == params.qF;
                q_cvx(1,:) >= params.q_min_xy(1); q_cvx(1,:) <= params.q_max_xy(1);
                q_cvx(2,:) >= params.q_min_xy(2); q_cvx(2,:) <= params.q_max_xy(2);
                q_cvx(3,:) == params.H_A;

                for n_c=1:N
                    sum_square(q_cvx(1:2,n_c) - params.u_R_pos(1:2)) + square(params.H_A - params.H_R) <= u_cvx(n_c);
                    sum_square(q_cvx(:,n_c) - params.u_s_3d) <= z_cvx(n_c);
                    for k_c = 1:K
                       sum_square(q_cvx(:,n_c) - params.u_k_coords_3d(k_c,:)') <= t_k_cvx(k_c,n_c);
                    end
                end

                % --- EFFICIENT TRUST REGION CONSTRAINT ---
                delta_q_per_slot = 5; % Allow each point to move up to 5 meters from its previous iteration's position
                for n_tr = 1:N
                    norm(q_cvx(:, n_tr) - q_iter(:, n_tr)) <= delta_q_per_slot;
                end
        cvx_end
        
        if contains(cvx_status, 'Solved')
            fprintf('      CVX Status: %s, Optimal value: %f\n', cvx_status, cvx_optval);
            if abs(cvx_optval - obj_val_prev_sca) / (abs(obj_val_prev_sca) + 1e-9) < epsilon_sca_conv
                fprintf('    SCA for Subproblem 2 converged.\n');
                q_iter = q_cvx; break;
            end
            q_iter = q_cvx; obj_val_prev_sca = cvx_optval;
        else
            fprintf('    Subproblem 2 CVX failed. Status: %s. Aborting SCA.\n', cvx_status);
            break;
        end
    end 
    q_opt = q_iter;
end










































































































% % solve_subproblem2_Q.m
% 
% function q_opt = solve_subproblem2_Q(params, s_k_fixed, W_k_fixed, q_initial)
%     % Solves Subproblem 2: UAV Trajectory (Q) Optimization using SCA.
%     
%     % Unpack parameters
%     N = params.N; K = params.K;
%     I_max_sca_local = params.I_max_sca_sub2;
%     epsilon_sca_conv = params.epsilon_sca;
%     
%     % Initialize SCA iteration
%     q_iter = q_initial;
%     obj_val_prev_sca = -Inf;
% 
%     fprintf('    Solving Subproblem 2 (Q) using SCA...\n');
% 
%     for sca_iter_idx = 1:I_max_sca_local
%         fprintf('      SCA Iteration %d for Subproblem 2\n', sca_iter_idx);
%         
%         % --- STEP 1: Pre-compute all SCA coefficients ---
%         % These are based on the current trajectory iterate, q_iter.
%         coeffs = precompute_sca_coeffs_for_Q(params, q_iter, s_k_fixed, W_k_fixed);
%         [u_iter, z_iter, t_k_iter] = calculate_slack_variables(params, q_iter);
% 
%         % --- STEP 2: Solve the convex optimization problem ---
%         cvx_begin
%             cvx_solver mosek;
%             
%             % --- CVX Variables ---
%             variable q_cvx(3, N)
%             variable t_k_cvx(K, N) nonnegative % Slack variable for ||q-uk||^2
%             variable u_cvx(1, N) nonnegative   % Slack variable for ||q-uR||^2
%             variable z_cvx(1, N) nonnegative   % Slack variable for ||q-us||^2
% 
%             % --- Build Objective Function Expressions ---
%             expression comm_rate_lb_total;
%             expression sensing_snr_lb_total;
% 
%             % Build the expressions using loops for clarity and to avoid dimension errors
%             comm_rate_lb_total = 0;
%             sensing_snr_lb_total = 0;
%             for n = 1:N
%                 sensing_snr_lb_n = coeffs.SNRs_r2(n) + ...
%                                    coeffs.A1_n(n) * (u_cvx(n) - u_iter(n)) + ...
%                                    coeffs.A2_n(n) * (z_cvx(n) - z_iter(n));
%                 sensing_snr_lb_total = sensing_snr_lb_total + sensing_snr_lb_n;
%                 
%                 for k = 1:K
%                     if s_k_fixed(k, n) == 1
%                         comm_rate_lb_kn = coeffs.Rk_r2(k, n) + ...
%                                           coeffs.R1_n(k, n) * (t_k_cvx(k, n) - t_k_iter(k, n)) + ...
%                                           coeffs.R2_n(k, n) * (u_cvx(n) - u_iter(n));
%                         comm_rate_lb_total = comm_rate_lb_total + comm_rate_lb_kn;
%                     end
%                 end
%             end
% 
%             maximize(params.beta_C * (comm_rate_lb_total/N) + (1-params.beta_C) * (sensing_snr_lb_total/N));
% 
%             % --- Constraints ---
%             subject to
%                 % ================================================================
%                 % === DEBUGGING: Temporarily disable the hard QoS constraints ===
%                 % ================================================================
%                 
%                 % QoS constraints C4 (Sensing)
%                 % sensing_snr_lb_total / N >= params.Gamma_SNR_linear;
%                 
%                 % QoS constraints C5 (Communication)
%                 % for k_c = 1:K
%                 %     user_k_rate_lb = 0;
%                 %     for n_c = 1:N
%                 %         if s_k_fixed(k_c, n_c) == 1
%                 %              user_k_rate_lb = user_k_rate_lb + ( ...
%                 %                  coeffs.Rk_r2(k_c, n_c) + ...
%                 %                  coeffs.R1_n(k_c, n_c) * (t_k_cvx(k_c, n_c) - t_k_iter(k_c, n_c)) + ...
%                 %                  coeffs.R2_n(k_c, n_c) * (u_cvx(n_c) - u_iter(n_c)) ...
%                 %              );
%                 %         end
%                 %     end
%                 %     user_k_rate_lb / N >= params.R_min;
%                 % end
%                 
%                 % ================================================================
% 
%                 % Mobility constraints C6, C7, C8, C9
%                 for n_c = 1:N-1
%                     norm(q_cvx(:, n_c+1) - q_cvx(:, n_c)) <= params.V_max * params.tau;
%                 end
%                 q_cvx(:, 1) == params.q0;
%                 q_cvx(:, N) == params.qF;
%                 q_cvx(1,:) >= params.q_min_xy(1); q_cvx(1,:) <= params.q_max_xy(1);
%                 q_cvx(2,:) >= params.q_min_xy(2); q_cvx(2,:) <= params.q_max_xy(2);
%                 q_cvx(3,:) == params.H_A;
% 
%                 % Slack variable constraints C11, C12, C13
%                 % These define the relationship between the trajectory q_cvx and
%                 % the slack variables u_cvx, z_cvx, and t_k_cvx.
%                 for n_c=1:N
%                     sum_square(q_cvx(1:2,n_c) - params.u_R_pos(1:2)) + square(params.H_A - params.H_R) <= u_cvx(n_c); % C11
%                     sum_square(q_cvx(:,n_c) - params.u_s_3d) <= z_cvx(n_c); % C12
%                     for k_c = 1:K
%                        sum_square(q_cvx(:,n_c) - params.u_k_coords_3d(k_c,:)') <= t_k_cvx(k_c,n_c); % C13
%                     end
%                 end
% 
% 
%                 % ================================================================
%                 % === TRUST REGION CONSTRAINT (THE FIX) ===
%                 % ================================================================
%                 % Limit how much the new trajectory can change from the previous iteration.
%                 % 'delta_q' is a tuning parameter. It can be related to V_max.
%                 delta_q = 5; % e.g., max change of 5 meters per time slot from previous plan
%                 for n_c_tr = 1:N
%                     norm(q_cvx(:, n_c_tr) - q_iter(:, n_c_tr)) <= delta_q;
%                 end
%         cvx_end
%         
%         if contains(cvx_status, 'Solved')
%             fprintf('      CVX Status: %s, Optimal value: %f\n', cvx_status, cvx_optval);
%             if abs(cvx_optval - obj_val_prev_sca) / (abs(obj_val_prev_sca) + 1e-9) < epsilon_sca_conv
%                 fprintf('    SCA for Subproblem 2 converged.\n');
%                 q_iter = q_cvx; break;
%             end
%             q_iter = q_cvx; obj_val_prev_sca = cvx_optval;
%         else
%             fprintf('    Subproblem 2 CVX failed. Status: %s. Aborting SCA.\n', cvx_status);
%             break;
%         end
%     end 
%     q_opt = q_iter;
% end