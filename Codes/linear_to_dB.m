function dB_val = linear_to_dB(linear_val)
    if linear_val <= 1e-30 % Handle very small or non-positive inputs to avoid -Inf from log10
        dB_val = -300; % Represents a very small dB value (e.g., -300 dB)
    else
        dB_val = 10 * log10(linear_val);
    end
end