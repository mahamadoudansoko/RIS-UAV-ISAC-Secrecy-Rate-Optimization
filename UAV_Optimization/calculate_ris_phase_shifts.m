
% calculate_ris_phase_shifts.m
function phi_all_slots = calculate_ris_phase_shifts(params, q_traj)
    % Handles both Optimized (PS) and Random (RPS) modes.
    if isfield(params, 'phase_shift_mode') && strcmp(params.phase_shift_mode, 'random')
        phi_all_slots = exp(1j * 2 * pi * rand(params.N, params.M));
        return;
    end

    % Normal PS mode
    N = params.N; M = params.M;
    phi_all_slots = zeros(N, M);
    m_vector = (0:M-1)'; % Antenna element indices

    for n = 1:N
        q_n = q_traj(:, n);
        
        % Calculate Angles (Departure/Arrival)
        % Angle from UAV to RIS (AoD)
        [theta_AR, ~] = calculate_angles_3d(q_n, params.u_R_pos);
        
        % Angle from RIS to Communication User (AoA, averaged for simplicity)
        theta_RC_avg = 0;
        for k_user = 1:params.K
            [theta_k, ~] = calculate_angles_3d(params.u_R_pos, params.u_k_coords_3d(k_user,:)');
            theta_RC_avg = theta_RC_avg + theta_k;
        end
        theta_RC_avg = theta_RC_avg / params.K;
        
        % Angle from RIS to Sensing Target (AoA)
        [theta_RS, ~] = calculate_angles_3d(params.u_R_pos, params.u_s_3d);
        
        % Phase shift required to align for communication
        phi_comm_align = - (theta_AR + theta_RC_avg);
        
        % Phase shift required to align for sensing
        phi_sense_align = - (theta_AR + theta_RS);
        
        % Weighted sum of the required phase shifts (as per Eq. 7)
        beta_C = params.beta_C;
        beta_S = 1 - beta_C;
        
        phi_n_rad = beta_C * (m_vector * pi * sin(phi_comm_align)) + ...
                    beta_S * (m_vector * pi * sin(phi_sense_align));
                    
        phi_all_slots(n, :) = exp(1j * phi_n_rad);
    end
end








































% function phi_vectors = calculate_ris_phase_shifts(params, q_trajectory)
%     % Calculates the RIS phase shifts for all N time slots.
%     % Handles both Optimized (PS) and Random (RPS) modes.
%     %
%     % Output: phi_vectors (N x M matrix of phase values in radians)
% 
%     % --- Handle the Random Phase Shift (RPS) baseline ---
%     if isfield(params, 'phase_shift_mode') && strcmp(params.phase_shift_mode, 'random')
%         fprintf('    Mode: Generating fixed random phase shifts for RPS baseline.\n');
%         rng(0); % for reproducibility
%         phi_vectors = 2 * pi * rand(params.N, params.M);
%         return; % Exit the function early
%     end
% 
%     % --- Normal Operation: Calculate Optimized Phase Shifts (PS) ---
%     N = params.N;
%     M = params.M;
% 
%     % Safety check: if q_trajectory is a single 3x1 vector
%     if size(q_trajectory, 2) == 1
%         N = 1;
%     end
% 
%     phi_vectors = zeros(N, M);
% 
%     for n = 1:N
%         if size(q_trajectory, 2) == 1
%             q_n = q_trajectory;  % Single vector case (3x1)
%         else
%             q_n = q_trajectory(:, n);   % Full trajectory case (3xN)
%         end
% 
%         AP_pos = params.qF;
%         RIS_pos = params.u_R_pos;
%         K = params.K;
% 
%         % UAV to AP (theta_AR)
%         [theta_AR, ~] = calculate_angles_3d(q_n, AP_pos);
% 
%         % UAV to RIS (theta_RS)
%         [theta_RS, ~] = calculate_angles_3d(q_n, RIS_pos);
% 
%         % RIS to Users
%         theta_k_RC_all = zeros(1, K);
%         for k = 1:K
%             user_pos = params.u_k_coords_3d(k, :);
%             [theta_k_RC, ~] = calculate_angles_3d(RIS_pos, user_pos);
%             theta_k_RC_all(k) = theta_k_RC;
%         end
% 
%         % === Use average angle from both users ===
%         theta_k_RC_avg = mean(theta_k_RC_all);
% 
%         % Weighted sum for phase shifts
%         phi_k_component = theta_k_RC_avg + theta_AR;
%         phi_s_component = theta_RS + theta_AR;
% 
%         beta_C = params.beta_C;
%         beta_S = 1 - beta_C;
% 
%         m_vector = (0:M-1)';
%         phase_vec_s = m_vector * pi * phi_s_component;
%         phase_vec_k = m_vector * pi * phi_k_component;
% 
%         phi_vectors(n, :) = mod(beta_S * phase_vec_s + beta_C * phase_vec_k, 2 * pi);
%     end
% end
