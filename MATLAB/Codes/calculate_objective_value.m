% calculate_objective_value.m

function [obj_val, avg_comm_sum_rate, avg_sensing_snr_linear] = calculate_objective_value(params, q_traj, s_k_sched, W_k_beamf)
    % Calculates the objective function value for a given set of variables.
    % This function computes the TRUE value of the objective, not a lower bound.
    %
    % Inputs:
    %   params: Structure containing system parameters (including beta_C).
    %   q_traj: UAV trajectory (3 x N).
    %   s_k_sched: CU scheduling (K x N).
    %   W_k_beamf: DFRC-BF matrices (V x V x K x N).
    %
    % Outputs:
    %   obj_val: Total weighted objective value.
    %   avg_comm_sum_rate: Average communication sum-rate over N slots (bits/s/Hz).
    %   avg_sensing_snr_linear: Average sensing SNR (linear scale) over N slots.

    % --- Unpack Parameters ---
    N = params.N;
    K = params.K;
    M_ris = params.M;
    sigma_k_sq = params.sigma_k_sq_linear;
    sigma_s_sq = params.sigma_s_sq_linear;
    beta_C = params.beta_C;
    beta_S = 1 - beta_C;

    % --- Get fixed coordinates ---
    u_R_pos = params.u_R_pos;
    u_k_coords = params.u_k_coords_3d;
    u_s_pos = params.u_s_3d;

    % --- Initialize Accumulators ---
    total_comm_rate = 0;
    total_sensing_snr = 0;

    % Calculate the phase shifts for the entire trajectory once.
    % This is more efficient than calling it inside the loop.
    phi_all_slots = calculate_ris_phase_shifts(params, q_traj);

    % --- Loop over all time slots to calculate performance ---
    for n = 1:N
        q_n = q_traj(:, n);
        s_k_n = s_k_sched(:, n);
        W_k_n_all = W_k_beamf(:, :, :, n);
        
        % Get the 1xM phase shift vector for this slot
        phi_n_vector = phi_all_slots(n, :);
        % Create the required MxM diagonal matrix from the vector
        Phi_n_matrix = diag(phi_n_vector);

        % Calculate channels for the current position
        [H_AR_n, h_k_AC_all, h_AS_n, h_k_RC_all, h_RS_n] = ...
            calculate_channels(q_n, u_R_pos, u_k_coords, u_s_pos, params);

        % --- Communication Rate Calculation for slot n ---
        slot_sum_rate = 0;
        for k = 1:K
            % Only calculate for the scheduled user
            if s_k_n(k) == 1
                % Start with the direct path channel
                h_eff_comm = h_k_AC_all(:, k);
                
                % If RIS exists, add the reflected path component
                if M_ris > 0
                    h_eff_comm = h_eff_comm + H_AR_n' * Phi_n_matrix * h_k_RC_all(:, k);
                end
                
                % Create the effective channel matrix H_k = h_k * h_k'
                H_eff_comm_matrix = h_eff_comm * h_eff_comm';
                W_k_n_user = W_k_n_all(:, :, k);
                
                % Calculate the received signal power
                signal_power = real(trace(H_eff_comm_matrix * W_k_n_user));
                
                % Calculate the achievable rate
                snr_k = max(0, signal_power) / sigma_k_sq;
                slot_sum_rate = slot_sum_rate + log2(1 + snr_k);
            end
        end
        total_comm_rate = total_comm_rate + slot_sum_rate;

        % --- Sensing SNR Calculation for slot n ---
        % Start with the direct sensing path
        h_eff_sense = h_AS_n;
        
        % If RIS exists, add the reflected sensing path
        if M_ris > 0
            h_eff_sense = h_eff_sense + H_AR_n' * Phi_n_matrix * h_RS_n;
        end
        
        % H_s in the paper is h_eff_sense * h_eff_sense'
        H_eff_sense_matrix = h_eff_sense * h_eff_sense';
        % The term in the SNR formula is H_s' * H_s
        H_sensing_term = H_eff_sense_matrix' * H_eff_sense_matrix;
        
        % The total transmitted power matrix for this slot
        W_total_n = sum(W_k_n_all, 3);
        
        % Calculate the received echo signal power
        sensing_signal_power = real(trace(H_sensing_term * W_total_n));
        
        % Calculate the sensing SNR
        slot_sensing_snr = max(0, sensing_signal_power) / sigma_s_sq;
        total_sensing_snr = total_sensing_snr + slot_sensing_snr;
    end

    % --- Final Averaged Metrics and Objective Value ---
    avg_comm_sum_rate = total_comm_rate / N;
    avg_sensing_snr_linear = total_sensing_snr / N;
    
    % Weighted sum objective
    obj_val = beta_S * avg_sensing_snr_linear + beta_C * avg_comm_sum_rate;
end