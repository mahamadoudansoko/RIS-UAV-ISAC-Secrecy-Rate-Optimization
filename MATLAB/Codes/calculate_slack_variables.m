% calculate_slack_variables.m
function [u, z, t_k] = calculate_slack_variables(params, q_traj)
    % Helper to calculate slack variable values from a given trajectory
    N = params.N; K = params.K;
    u = zeros(1, N);
    z = zeros(1, N);
    t_k = zeros(K, N);
    for n = 1:N
        % C11 relation
        u(n) = sum_square(q_traj(1:2, n) - params.u_R_pos(1:2)) + (params.H_A - params.H_R)^2;
        % C12 relation
        z(n) = sum_square(q_traj(:, n) - params.u_s_3d);
        % C13 relation
        for k = 1:K
            t_k(k, n) = sum_square(q_traj(:, n) - params.u_k_coords_3d(k,:)');
        end
    end
end