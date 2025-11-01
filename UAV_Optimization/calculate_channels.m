

% calculate_channels.m
function [H_AR_n, h_k_AC_all, h_AS_n, h_k_RC_all, h_RS_n] = calculate_channels(q_n, u_R_pos, u_k_coords_all_3d, u_s_pos_3d, params)
    % Unpack parameters
    V = params.V; M = params.M; K = params.K;
    beta_0 = params.beta_0_linear;
    lambda_c = params.lambda_c;

    % --- UAV to RIS Channel (LoS) ---
    dist_AR = norm(q_n - u_R_pos);
    path_loss_AR = sqrt(beta_0 / (dist_AR ^ params.alpha_AR));
    [theta_AR_tx, ~] = calculate_angles_3d(q_n, u_R_pos);
    [theta_AR_rx, ~] = calculate_angles_3d(u_R_pos, q_n);
    a_tx = steering_vector_ula(V, lambda_c/2, lambda_c, theta_AR_tx);
    a_rx = steering_vector_ula(M, lambda_c/2, lambda_c, theta_AR_rx);
    H_AR_n = path_loss_AR * (a_rx * a_tx'); % MxV matrix

    % --- UAV to CU Channels (LoS) ---
    h_k_AC_all = zeros(V, K);
    for k = 1:K
        u_k_pos = u_k_coords_all_3d(k,:)';
        dist_AC = norm(q_n - u_k_pos);
        path_loss_AC = sqrt(beta_0 / (dist_AC ^ 2.2)); % Standard LoS exponent
        [theta_AC, ~] = calculate_angles_3d(q_n, u_k_pos);
        h_k_AC_all(:, k) = path_loss_AC * steering_vector_ula(V, lambda_c/2, lambda_c, theta_AC);
    end

    % --- UAV to Target Channel (LoS) ---
    dist_AS = norm(q_n - u_s_pos_3d);
    path_loss_AS = sqrt(beta_0 / (dist_AS ^ params.alpha_RS));
    [theta_AS, ~] = calculate_angles_3d(q_n, u_s_pos_3d);
    h_AS_n = path_loss_AS * steering_vector_ula(V, lambda_c/2, lambda_c, theta_AS);

    % --- RIS to CU Channels (Rician) ---
    h_k_RC_all = zeros(M, K);
    kappa_RC = params.kappa_k_RC_linear;
    for k = 1:K
        u_k_pos = u_k_coords_all_3d(k,:)';
        dist_RC = norm(u_R_pos - u_k_pos);
        path_loss_RC = sqrt(beta_0 / (dist_RC ^ params.alpha_k_RC));
        [theta_RC, ~] = calculate_angles_3d(u_R_pos, u_k_pos);
        h_los = steering_vector_ula(M, lambda_c/2, lambda_c, theta_RC);
        h_nlos = (randn(M, 1) + 1j * randn(M, 1)) / sqrt(2);
        h_k_RC_all(:, k) = path_loss_RC * (sqrt(kappa_RC / (kappa_RC + 1)) * h_los + sqrt(1 / (kappa_RC + 1)) * h_nlos);
    end
    
    % --- RIS to Target Channel (Rician) ---
    kappa_RS = params.kappa_RS_linear;
    dist_RS = norm(u_R_pos - u_s_pos_3d);
    path_loss_RS = sqrt(beta_0 / (dist_RS ^ params.alpha_RS));
    [theta_RS, ~] = calculate_angles_3d(u_R_pos, u_s_pos_3d);
    h_los_s = steering_vector_ula(M, lambda_c/2, lambda_c, theta_RS);
    h_nlos_s = (randn(M, 1) + 1j * randn(M, 1)) / sqrt(2);
    h_RS_n = path_loss_RS * (sqrt(kappa_RS / (kappa_RS + 1)) * h_los_s + sqrt(1 / (kappa_RS + 1)) * h_nlos_s);
end

function a = steering_vector_ula(num_ant, d, lambda, theta)
    % Creates a steering vector for a uniform linear array
    m = (0:num_ant-1)';
    a = exp(-1j * 2 * pi * d * m * sin(theta) / lambda);
end



























% % calculate_channels.m
% function [H_AR_n, h_k_AC_n_all, h_AS_n, h_k_RC_n_all, h_RS_n, ...
%           h_AR_steer_tx_n, h_RA_steer_rx_n, h_k_AC_steer_all_n, h_AS_steer_n, ...
%           h_k_RC_LoS_steer_all_n, h_RS_LoS_steer_n] = ...
%           calculate_channels(q_n, u_R_pos, u_k_coords_all_3d, u_s_pos_3d, params)
%     q_n = q_n(:);             % Force 3x1
%     u_R_pos = u_R_pos(:);     % Force 3x1
%     u_s_pos_3d = u_s_pos_3d(:); % Force 3x1
% 
% 
%     if size(u_k_coords_all_3d, 2) ~= 3
%     error('u_k_coords_all_3d must be of size K x 3 (K users, 3D each).');
%     end
% 
%     % fprintf('--- Inside calculate_channels.m ---\n'); % Can be commented out for final runs
%     
%     % Parameters
%     V = params.V; M = params.M; K = params.K;
%     lambda_c = params.lambda_c; beta_0_linear = params.beta_0_linear;
%     alpha_RC_exp = params.alpha_k_RC; alpha_RS_exp = params.alpha_RS; alpha_AR_exp = params.alpha_AR; 
%     kappa_k_RC_linear = params.kappa_k_RC_linear; kappa_RS_linear = params.kappa_RS_linear;
%     d_UAV_element_spacing = params.lambda_c / 2; d_RIS_element_spacing = params.lambda_c / 2;
% 
%     % Initialize outputs for full channels
%     H_AR_n = zeros(M, V);
%     h_k_AC_n_all = zeros(V, K);
%     h_AS_n = zeros(V, 1);
%     h_k_RC_n_all = zeros(M, K);
%     h_RS_n = zeros(M, 1);
% 
%     % Initialize outputs for steering vectors (angular parts)
%     h_AR_steer_tx_n = zeros(V,1);        % UAV Tx steering to RIS
%     h_RA_steer_rx_n = zeros(M,1);        % RIS Rx steering from UAV
%     h_k_AC_steer_all_n = zeros(V, K);    % UAV Tx steering to CUs
%     h_AS_steer_n = zeros(V,1);           % UAV Tx steering to Target
%     h_k_RC_LoS_steer_all_n = zeros(M,K); % RIS Tx LoS steering to CUs
%     h_RS_LoS_steer_n = zeros(M,1);       % RIS Tx LoS steering to Target
% 
%     if M > 0 && V > 0 
%         dist_UAV_RIS = calculate_distance(q_n, u_R_pos);
%         dist_UAV_RIS = max(1e-3, dist_UAV_RIS); % Avoid dist = 0
%         path_loss_UAV_RIS = sqrt(beta_0_linear / (dist_UAV_RIS^alpha_AR_exp));
%         
%         [elev_UAV_tx_to_RIS, ~] = calculate_angles_3d(q_n, u_R_pos);       
%         [elev_RIS_rx_from_UAV, ~] = calculate_angles_3d(u_R_pos, q_n);   
%     
%         h_AR_steer_tx_n = steering_vector_ula(V, d_UAV_element_spacing, lambda_c, elev_UAV_tx_to_RIS); 
%         h_RA_steer_rx_n = steering_vector_ula(M, d_RIS_element_spacing, lambda_c, elev_RIS_rx_from_UAV); 
%         
%         H_AR_n = path_loss_UAV_RIS * (h_RA_steer_rx_n * h_AR_steer_tx_n'); 
%     end
% 
%     if V > 0
%         for k_idx = 1:K
%             u_k_pos = u_k_coords_all_3d(k_idx, :)'; 
%             dist_UAV_CUk = calculate_distance(q_n, u_k_pos);
%             dist_UAV_CUk = max(1e-3, dist_UAV_CUk);
%             path_loss_UAV_CUk = sqrt(beta_0_linear / (dist_UAV_CUk^alpha_RC_exp));
%             [elev_UAV_to_CUk, ~] = calculate_angles_3d(q_n, u_k_pos);
%             
%             h_k_AC_steer_all_n(:, k_idx) = steering_vector_ula(V, d_UAV_element_spacing, lambda_c, elev_UAV_to_CUk);
%             h_k_AC_n_all(:, k_idx) = path_loss_UAV_CUk * h_k_AC_steer_all_n(:, k_idx);
%         end
%     end
% 
%     if V > 0
%         dist_UAV_Target = calculate_distance(q_n, u_s_pos_3d);
%         dist_UAV_Target = max(1e-3, dist_UAV_Target);
%         path_loss_UAV_Target = sqrt(beta_0_linear / (dist_UAV_Target^alpha_RS_exp));
%         [elev_UAV_to_Target, ~] = calculate_angles_3d(q_n, u_s_pos_3d);
%         
%         h_AS_steer_n = steering_vector_ula(V, d_UAV_element_spacing, lambda_c, elev_UAV_to_Target);
%         h_AS_n = path_loss_UAV_Target * h_AS_steer_n;
%     end
% 
%     if M > 0
%         for k_idx = 1:K
%             u_k_pos = u_k_coords_all_3d(k_idx, :)';
%             dist_RIS_CUk = calculate_distance(u_R_pos, u_k_pos);
%             dist_RIS_CUk = max(1e-3, dist_RIS_CUk);
%             path_loss_RIS_CUk = sqrt(beta_0_linear / (dist_RIS_CUk^alpha_RC_exp));
%             [elev_RIS_to_CUk, ~] = calculate_angles_3d(u_R_pos, u_k_pos); 
%             
%             h_k_RC_LoS_steer_all_n(:, k_idx) = steering_vector_ula(M, d_RIS_element_spacing, lambda_c, elev_RIS_to_CUk);
%             hat_g_RC = randn_complex(M, 1); % NLOS component
%             
%             h_k_RC_n_all(:, k_idx) = path_loss_RIS_CUk * ...
%                 (sqrt(kappa_k_RC_linear / (1 + kappa_k_RC_linear)) * h_k_RC_LoS_steer_all_n(:, k_idx) + ...
%                  sqrt(1 / (1 + kappa_k_RC_linear)) * hat_g_RC); 
%         end
%     end
% 
%     if M > 0
%         dist_RIS_Target = calculate_distance(u_R_pos, u_s_pos_3d);
%         dist_RIS_Target = max(1e-3, dist_RIS_Target);
%         path_loss_RIS_Target = sqrt(beta_0_linear / (dist_RIS_Target^alpha_RS_exp));
%         [elev_RIS_to_Target, ~] = calculate_angles_3d(u_R_pos, u_s_pos_3d); 
%         
%         h_RS_LoS_steer_n = steering_vector_ula(M, d_RIS_element_spacing, lambda_c, elev_RIS_to_Target);
%         hat_g_RS = randn_complex(M, 1); % NLOS component
%         
%         h_RS_n = path_loss_RIS_Target * ...
%             (sqrt(kappa_RS_linear / (1 + kappa_RS_linear)) * h_RS_LoS_steer_n + ...
%              sqrt(1 / (1 + kappa_RS_linear)) * hat_g_RS); 
%     end
% 
%     % fprintf('--- Exiting calculate_channels.m ---\n'); % Can be commented out
% end