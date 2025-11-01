% calculate_final_metrics.m

function [avg_rate, avg_snr_db] = calculate_final_metrics(params, q_opt, s_k_opt, W_k_opt)
    % This function calculates the final, true performance metrics (Rate and SNR)
    % after the AO algorithm has converged. It uses the same logic as
    % calculate_objective_value.m but returns the separate metrics.

    % Unpack Parameters
    N = params.N;
    K = params.K;
    M_ris = params.M;
    sigma_k_sq = params.sigma_k_sq_linear;
    sigma_s_sq = params.sigma_s_sq_linear;

    % Get fixed coordinates from the params struct
    u_R_pos = params.u_R_pos;
    u_k_coords = params.u_k_coords_3d;
    u_s_pos = params.u_s_3d;
    
    % Initialize Accumulators
    total_comm_rate = 0;
    total_sensing_snr_linear = 0;

    % Calculate the phase shifts for the entire optimized trajectory
    phi_all_slots = calculate_ris_phase_shifts(params, q_opt);

    for n = 1:N
        q_n = q_opt(:, n);
        s_k_n = s_k_opt(:, n);
        W_k_n_all = W_k_opt(:, :, :, n);
        
        % Create the diagonal MxM phase shift matrix for this slot
        Phi_n_matrix = diag(phi_all_slots(n, :));

        % --- THIS IS THE CORRECTED FUNCTION CALL ---
        % Provide all 5 required arguments to calculate_channels
        [H_AR_n, h_k_AC_all, h_AS_n, h_k_RC_all, h_RS_n] = ...
            calculate_channels(q_n, u_R_pos, u_k_coords, u_s_pos, params);

        % --- Calculate Communication Rate for slot n ---
        slot_sum_rate = 0;
        for k = 1:K
            if s_k_n(k) == 1
                h_eff_comm = h_k_AC_all(:, k);
                if M_ris > 0
                    h_eff_comm = h_eff_comm + H_AR_n' * Phi_n_matrix * h_k_RC_all(:, k);
                end
                H_eff_comm_matrix = h_eff_comm * h_eff_comm';
                W_k_n_user = W_k_n_all(:, :, k);
                signal_power = real(trace(H_eff_comm_matrix * W_k_n_user));
                snr_k = max(0, signal_power) / sigma_k_sq;
                slot_sum_rate = slot_sum_rate + log2(1 + snr_k);
            end
        end
        total_comm_rate = total_comm_rate + slot_sum_rate;

        % --- Calculate Sensing SNR for slot n ---
        h_eff_sense = h_AS_n;
        if M_ris > 0
            h_eff_sense = h_eff_sense + H_AR_n' * Phi_n_matrix * h_RS_n;
        end
        H_eff_sense_matrix = h_eff_sense * h_eff_sense';
        H_sensing_term = H_eff_sense_matrix' * H_eff_sense_matrix;
        W_total_n = sum(W_k_n_all, 3);
        sensing_signal_power = real(trace(H_sensing_term * W_total_n));
        slot_sensing_snr = max(0, sensing_signal_power) / sigma_s_sq;
        total_sensing_snr_linear = total_sensing_snr_linear + slot_sensing_snr;
    end

    % --- Calculate and return the final averaged metrics ---
    avg_rate = total_comm_rate / N;
    avg_snr_linear = total_sensing_snr_linear / N;
    
    % Convert final sensing SNR to dB for plotting
    if avg_snr_linear > 0
        avg_snr_db = 10 * log10(avg_snr_linear);
    else
        avg_snr_db = -Inf; % Or a very small number like -100
    end
end