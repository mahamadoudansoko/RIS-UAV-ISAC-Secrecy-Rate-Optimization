function h = steering_vector_ula(num_elements, element_spacing, wavelength, angle_rad)
    % Generates a steering vector for a Uniform Linear Array (ULA).
    % Assumes elements are spaced along an axis, and angle_rad is the
    % effective angle of departure/arrival such that sin(angle_rad) is the
    % phase progression factor along the array.
    
    h = zeros(num_elements, 1);
    if num_elements > 0
        m_indices = (0:num_elements-1)'; % 0-indexed for simpler formula m*...
        phase_progression_factor = 2 * pi * element_spacing / wavelength;
        h = exp(1j * m_indices * phase_progression_factor * sin(angle_rad));
    end
end