function dist = calculate_distance(pos1, pos2)
    pos1 = pos1(:);  % Ensure column
    pos2 = pos2(:);  % Ensure column
    dist = norm(pos1 - pos2);
end