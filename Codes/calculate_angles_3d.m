function [elevation_rad, azimuth_rad] = calculate_angles_3d(source_pos, dest_pos)
    % Calculates elevation and azimuth angles from source to destination
    % source_pos, dest_pos are 3D column vectors [x; y; z]
    
    vec = dest_pos - source_pos;
    
    % Azimuth angle (angle in XY plane from positive X-axis, counter-clockwise)
    % Range: -pi to pi
    azimuth_rad = atan2(vec(2), vec(1));
    
    % Elevation angle (angle from XY plane up to Z-axis)
    % Range: -pi/2 to pi/2
    horizontal_dist = norm(vec(1:2));
    if horizontal_dist < eps % Avoid division by zero or issues if vec(1) and vec(2) are zero
        if vec(3) > 0
            elevation_rad = pi/2;
        elseif vec(3) < 0
            elevation_rad = -pi/2;
        else
            elevation_rad = 0; % Or NaN, as direction is undefined if source_pos == dest_pos
        end
    else
        elevation_rad = atan2(vec(3), horizontal_dist);
    end
end