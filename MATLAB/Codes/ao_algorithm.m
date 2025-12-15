% ao_algorithm.m
function [q_opt, s_k_opt, W_k_opt, phi_opt] = ao_algorithm(params, initial_q, initial_s_k, initial_W_k)
    % Implements the Alternating Optimization (AO) algorithm with explicit data flow.
    
    N = params.N; K = params.K;
    epsilon_ao = params.epsilon_ao; I_max = params.I_max_ao;
    
    % --- Initialize variables ---
    q_current = initial_q;
    s_k_current = initial_s_k;
    W_k_current = initial_W_k;
    
%     % EXPLICITLY calculate initial phase shifts based on initial trajectory
%     phi_current = calculate_ris_phase_shifts(params, q_current);
    
    obj_values = [];
    r3 = 1;

    fprintf('Starting AO Algorithm for beta_C = %.2f\n', params.beta_C);

    while r3 <= I_max
        fprintf('  AO Iteration %d\n', r3);

        % --- Step 1: Solve for S and W, given fixed q and phi ---
        [s_k_next_cont, W_k_next] = solve_subproblem1_S_W(q_current, s_k_current, W_k_current, params);

        s_k_next = zeros(K, N);

        for n=1:N
            [max_val,idx]=max(s_k_next_cont(:,n));
            if max_val > 0.5 % A simple rounding threshold
                s_k_next(idx,n)= 1;
            end
        end

        
        % --- Step 2: Solve for Q, given fixed S and W ---
        q_next = q_current; % Default to current if trajectory is fixed

        if ~(isfield(params, 'trajectory_mode') && strcmp(params.trajectory_mode, 'fixed'))
            % Pass the newly optimized S and W to the trajectory solver
            q_next = solve_subproblem2_Q(params, s_k_next, W_k_next, q_current);
        else
            fprintf('    Trajectory q fixed for SF baseline.\n');
        end
        

      % --- Step 3: Update phase shifts for the new trajectory ---
        %  Remove extra phi_next argument
        current_obj_val = calculate_objective_value(params, q_next, s_k_next, W_k_next);
        obj_values = [obj_values, current_obj_val];
        fprintf('    Current Solver Objective Value: %.4f\n', current_obj_val);


        %Check for convergence
        if r3>1 && abs(obj_values(end)-obj_values(end-1))/(abs(obj_values(end-1))+1e-9) < epsilon_ao
            fprintf('  AO Algorithm converged at iteration %d.\n', r3);
            break;
        end

        % Update all variables for the next iteration
        q_current = q_next;
        s_k_current = s_k_next;
        W_k_current = W_k_next;


        r3 = r3 + 1;
    end

    q_opt = q_current;
    s_k_opt = s_k_current;
    W_k_opt = W_k_current;
    
    % Calculate the final optimal phase shifts based on the final optimal trajectory
    phi_opt = calculate_ris_phase_shifts(params, q_opt); % Final phase shifts based on final trajectory

    if r3 > I_max, fprintf('AO Algorithm reached max iterations (%d).\n', I_max); end
    fprintf('AO Algorithm finished.\n');
end
