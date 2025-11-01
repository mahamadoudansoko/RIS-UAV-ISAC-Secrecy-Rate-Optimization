function linear_val = dB_to_linear(dB_val)
    % Converts a value from dB to linear scale.
    linear_val = 10.^(dB_val / 10);
end