% main_script_for_figures.m
clear;
close all;
clc;

script_path = fileparts(mfilename('fullpath'));
if isempty(script_path), script_path = pwd; end
addpath(genpath(script_path));
fprintf('Added to path: %s\n', script_path);

% --- System Parameters (TUNED FOR SPEED) ---
params.V = 10;       % Reduced from 25 for speed
params.M = 40;      % Reduced from 100 for speed
params.K = 2;
params.N = 40;      % Reduced from 50/100 for speed
params.tau = 0.5;
params.T = params.N * params.tau;
params.lambda_c = 0.1;
params.beta_0_dBW = -30;
params.P_UAV_dBm = 36;
params.V_max = 20;
params.H_A = 100;
params.H_R = 15;
params.alpha_k_RC = 2.5;
params.alpha_RS = 2.5;
params.alpha_AR = 2.2;
params.sigma_s_sq_dBm = -114;
params.sigma_k_sq_dBm = -114;
params.kappa_k_RC_dB = 3;
params.kappa_RS_dB = 3;
params.R_min = 1.0; % Using relaxed QoS for stability
params.Gamma_SNR_dB = 2.0; % Using relaxed QoS for stability
params.q0 = [0; 150; params.H_A];
params.qF = [300; 150; params.H_A];
params.q_min_xy = [0; 0];
params.q_max_xy = [300; 300];




% Locations
params.u_R_pos = [150; 90; params.H_R];
params.u_s_3d  = [60; 235; 0];
u_k_xy_temp = [230, 250; 250, 230];
params.u_k_coords_3d = [u_k_xy_temp, zeros(params.K, 1)];

% Conversions
params.beta_0_linear = 10^(params.beta_0_dBW/10);
params.P_UAV_linear = 10^((params.P_UAV_dBm-30)/10);
params.sigma_s_sq_linear = 10^((params.sigma_s_sq_dBm-30)/10);
params.sigma_k_sq_linear = 10^((params.sigma_k_sq_dBm-30)/10);
params.kappa_k_RC_linear = 10^(params.kappa_k_RC_dB/10);
params.kappa_RS_linear = 10^(params.kappa_RS_dB/10);
params.Gamma_SNR_linear = 10^(params.Gamma_SNR_dB/10);

% --- Algorithm Parameters (TUNED FOR SPEED) ---
params.epsilon_ao = 1e-2;
params.I_max_ao = 8;
params.I_max_sca_sub1 = 2;
params.I_max_sca_sub2 = 2;
params.epsilon_sca = 1e-3;

% --- Initializations ---
initial_q = zeros(3, params.N);
for n = 1:params.N, initial_q(:, n) = params.q0 + (params.qF - params.q0) * ((n-1) / max(1, params.N-1)); end
initial_s_k = zeros(params.K, params.N);
for n = 1:params.N, initial_s_k(mod(n-1, params.K) + 1, n) = 1; end
initial_W_k = zeros(params.V, params.V, params.K, params.N);
rng(0);
for n = 1:params.N
    if any(initial_s_k(:, n))
        power_share = params.P_UAV_linear / sum(initial_s_k(:, n));
        for k = 1:params.K
            if initial_s_k(k, n) == 1
                initial_W_k(:, :, k, n) = eye(params.V) * (power_share / params.V);
            end
        end
    end
end

% --- Simulation for Figures ---
simulation_flags.fig2 = true;
simulation_flags.fig3 = true;
simulation_flags.fig4 = true;

% --- Figure 2: UAV's Trajectories ---
if simulation_flags.fig2
    fprintf('--- Simulating Figure 2: UAV Trajectories ---\n');
    fig2_results = struct();
    beta_C_values_fig2 = [0, 0.5, 1];
    for i = 1:length(beta_C_values_fig2)
        beta_C_val = beta_C_values_fig2(i);
        current_params = params;
        current_params.beta_C = beta_C_val;
        fprintf('Running PS for beta_C = %.2f\n', beta_C_val);
        [q_opt, ~, ~, ~] = ao_algorithm(current_params, initial_q, initial_s_k, initial_W_k);
        fig2_results.(sprintf('PS_betaC_%s', strrep(num2str(beta_C_val), '.', '_'))).q_opt = q_opt;
    end

    fprintf('Running NR Baseline (NO-RIS) for Fig 2\n');
    params_NR_fig2 = params;
    params_NR_fig2.M = 0;
    params_NR_fig2.beta_C = 0.5;
    [q_opt_NR, ~, ~, ~] = ao_algorithm(params_NR_fig2, initial_q, initial_s_k, initial_W_k);
    fig2_results.NR_betaC_0_5.q_opt = q_opt_NR;
    
    fig2_results.plot_info = params; % Store all params for plotting
    save('fig2_results.mat', 'fig2_results');
    fprintf('Figure 2 simulation complete. Results saved to fig2_results.mat\n\n');
end

% --- Figure 3: Rate-SNR Regions ---
if simulation_flags.fig3
    fprintf('--- Simulating Figure 3: Rate-SNR Regions ---\n');
    fig3_results = struct();
    beta_C_range = 0:0.2:1; % Use a coarser range for speed
    schemes = {'PS', 'NR', 'RPS', 'SF'};
    for s_idx = 1:length(schemes)
        scheme = schemes{s_idx};
        fprintf('Running scheme: %s\n', scheme);
        rates = zeros(size(beta_C_range));
        snrs = zeros(size(beta_C_range));
        temp_params = params;
        
        for i = 1:length(beta_C_range)
            beta_C = beta_C_range(i);
            fprintf('  beta_C = %.2f\n', beta_C);
            current_params = temp_params;
            current_params.beta_C = beta_C;
            
            if strcmp(scheme, 'NR'), current_params.M = 0; end
            if strcmp(scheme, 'RPS'), current_params.phase_shift_mode = 'random'; end
            if strcmp(scheme, 'SF'), current_params.trajectory_mode = 'fixed'; end
            
            [q_opt, s_k_opt, W_k_opt, ~] = ao_algorithm(current_params, initial_q, initial_s_k, initial_W_k);
            [rates(i), snrs(i)] = calculate_final_metrics(current_params, q_opt, s_k_opt, W_k_opt);
        end
        fig3_results.(scheme).rates = rates;
        fig3_results.(scheme).snrs = snrs;
    end
    save('fig3_results.mat', 'fig3_results');
    fprintf('Figure 3 simulation complete. Results saved to fig3_results.mat\n\n');
end

% --- Figure 4: Performance vs. Number of RIS Elements ---
if simulation_flags.fig4
    fprintf('--- Simulating Figure 4: Performance vs. Number of RIS Elements ---\n');
    fig4_results = struct();
    M_range = 20:20:100; % Use a coarser range for speed
    schemes = {'PS', 'PS_RL1', 'PS_RL2'};
    for s_idx = 1:length(schemes)
        scheme = schemes{s_idx};
        fprintf('Running scheme: %s\n', scheme);
        rates = zeros(size(M_range));
        snrs = zeros(size(M_range));
        
        for i = 1:length(M_range)
            M_val = M_range(i);
            fprintf('  M = %d\n', M_val);
            current_params = params;
            current_params.M = M_val;
            current_params.beta_C = 0.5;
            
            if strcmp(scheme, 'PS_RL1'), current_params.u_R_pos = [70; 210; params.H_R]; end
            if strcmp(scheme, 'PS_RL2'), current_params.u_R_pos = [240; 240; params.H_R]; end
            
            [q_opt, s_k_opt, W_k_opt, ~] = ao_algorithm(current_params, initial_q, initial_s_k, initial_W_k);
            [rates(i), snrs(i)] = calculate_final_metrics(current_params, q_opt, s_k_opt, W_k_opt);
        end
        fig4_results.(scheme).rates = rates;
        fig4_results.(scheme).snrs = snrs;
    end
    fig4_results.M_range = M_range;
    save('fig4_results.mat', 'fig4_results');
    fprintf('Figure 4 simulation complete. Results saved to fig4_results.mat\n\n');
end

fprintf('All enabled simulations complete.\n');