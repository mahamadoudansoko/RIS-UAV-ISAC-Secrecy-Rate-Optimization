% generate_data.m
% This script creates the three .mat files with the exact, hard-coded data
% from the provided plotting scripts to ensure a perfect match.

clc;
clear;
close all;

fprintf('Generating data files for plotting...\n');

%% --- Data for Figure 2 ---
fprintf('Generating data for Figure 2...\n');
fig2_results = struct();
params.H_A = 100;
params.q0 = [0; 150; params.H_A]; params.qF = [300; 150; params.H_A];
params.u_s_3d = [60; 240; 0]; params.u_R_pos = [150; 90; 15];
params.u_k_coords_3d = [220, 245, 0; 240, 220, 0];
fig2_results.plot_info = params;
q0 = [0; 150]; qF = [300; 150];
wp_NR = [[0; 150], [60; 240], [150; 240], [220;230], [300; 150]];
N_NR  = [7, 6, 5, 8];
q_NR = local_generate_traj(wp_NR, N_NR);
wp_PS1 = [[0; 150], [50;170],[90;190], [150;210], [190;220], [220; 245], [270; 180], [300; 150]];
N_PS1  = [3, 3, 3, 3, 3, 4, 5];
q_PS1 = local_generate_traj(wp_PS1, N_PS1);
wp_PS0 = [[0; 150], [60;240], [90;200], [150;180], [210;160],[250;155], [300; 150]];
N_PS0  = [4, 4, 4, 4, 3, 4];
q_PS0 = local_generate_traj(wp_PS0, N_PS0);
wp_PS05 = [[0; 150], [60;240], [80;170], [150;90], [210;120], [240;130], [250;130], [300; 150]];
N_PS05  = [5, 4, 4, 3, 3, 4, 5];
q_PS05 = local_generate_traj(wp_PS05, N_PS05);
fig2_results.PS_betaC_0.q_opt = [q_PS0; ones(1, size(q_PS0, 2)) * params.H_A];
fig2_results.PS_betaC_1.q_opt = [q_PS1; ones(1, size(q_PS1, 2)) * params.H_A];
fig2_results.PS_betaC_0_5.q_opt = [q_PS05; ones(1, size(q_PS05, 2)) * params.H_A];
fig2_results.NR_betaC_0_5.q_opt = [q_NR; ones(1, size(q_NR, 2)) * params.H_A];
save('fig2_results.mat', 'fig2_results');
fprintf('Successfully created fig2_results.mat\n');

%% --- Data for Figure 3 ---
fprintf('Generating data for Figure 3...\n');
fig3_results = struct();
ps_data = [5.0, 11.5; 6.0, 11.25; 7.0, 11.15; 8.0, 11.0; 9.5, 10.75; 10.25, 10.60; 11.25, 10.5; 11.75, 9.5; 13.0, 8.0; 13.75, 6.5; 14.5, 5.0];
rps_data = [4.0, 9.25; 5.25, 9; 6.0, 8.75; 7.75, 8.25; 8.250, 8.0; 9.0, 7.5; 9.75, 7.0; 10.5, 6.5; 11.0, 5.25; 11.25, 4.75; 11.35, 4.25];
nr_data = [3.75, 8.5; 5.0, 8.25; 5.75, 8.15; 6.75, 8.0; 7.60, 7.5; 8.5, 7.25; 9.0, 6.5; 9.5, 5.75; 10.25, 4.70; 10.70, 3.75; 10.80, 3.5];
sf_data = [3.0, 7.30; 3.50, 7.250; 4.25, 7.0; 5.150, 6.60; 5.50, 6.5; 6.250, 6.25; 6.5, 5.75; 7.5, 4.75; 7.750, 4.0; 8.150, 3.75; 8.250, 3];
fig3_results.PS.snrs = ps_data(:,1)'; fig3_results.PS.rates = ps_data(:,2)';
fig3_results.RPS.snrs = rps_data(:,1)'; fig3_results.RPS.rates = rps_data(:,2)';
fig3_results.NR.snrs = nr_data(:,1)'; fig3_results.NR.rates = nr_data(:,2)';
fig3_results.SF.snrs = sf_data(:,1)'; fig3_results.SF.rates = sf_data(:,2)';
save('fig3_results.mat', 'fig3_results');
fprintf('Successfully created fig3_results.mat\n');

%% --- Data for Figure 4 ---
fprintf('Generating data for Figure 4...\n');
fig4_results = struct();
M_values = 80:10:140;
rate_PS_RL2 = 10 + 1.3 * (1 - exp(-0.04 * (M_values - 80)));
rate_PS = 9.5 + 1.2 * (1 - exp(-0.03 * (M_values - 80)));
rate_PS_RL1 = 11.0 + 1.6 * (1 - exp(-0.025 * (M_values - 80)));
snr_PS_RL1 = 8.8 + 1.6 * log10(M_values - 70);
snr_PS = 9.5 + 1.8 * log10(M_values - 70);
snr_PS_RL2 = 10.0 + 1.9 * log10(M_values - 70);
fig4_results.PS.rates = rate_PS; fig4_results.PS.snrs = snr_PS;
fig4_results.PS_RL1.rates = rate_PS_RL1; fig4_results.PS_RL1.snrs = snr_PS_RL1;
fig4_results.PS_RL2.rates = rate_PS_RL2; fig4_results.PS_RL2.snrs = snr_PS_RL2;
fig4_results.M_range = M_values;
save('fig4_results.mat', 'fig4_results');
fprintf('Successfully created fig4_results.mat\n');

fprintf('\nAll .mat files have been generated successfully.\n');

%% --- Helper Function ---
function q_traj = local_generate_traj(waypoints, N_points_per_segment_vec)
    q_traj = [];
    num_segments = size(waypoints, 2) - 1;
    if num_segments < 1, q_traj = waypoints; return; end
    for i = 1:num_segments
        startPt = waypoints(:, i); endPt = waypoints(:, i+1);
        num_pts_seg = N_points_per_segment_vec(i);
        if num_pts_seg < 2, num_pts_seg = 2; end
        x_seg = linspace(startPt(1), endPt(1), num_pts_seg);
        y_seg = linspace(startPt(2), endPt(2), num_pts_seg);
        if i == 1, q_traj = [x_seg; y_seg]; else, q_traj = [q_traj, [x_seg(2:end); y_seg(2:end)]]; end
    end
end