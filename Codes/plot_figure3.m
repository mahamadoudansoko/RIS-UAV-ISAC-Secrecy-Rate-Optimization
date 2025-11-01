% plot_figure3.m (Enhanced Design)
clc; clear; close all;

% --- Load Data ---
try
    load('fig3_results.mat');
    fprintf('Successfully loaded fig3_results.mat\n');
catch
    error('Could not find fig3_results.mat. Please run generate_data.m first.');
end

% --- Extract Data ---
ps_data  = [fig3_results.PS.snrs',  fig3_results.PS.rates'];
rps_data = [fig3_results.RPS.snrs', fig3_results.RPS.rates'];
nr_data  = [fig3_results.NR.snrs',  fig3_results.NR.rates'];
sf_data  = [fig3_results.SF.snrs',  fig3_results.SF.rates'];

% --- Setup Figure and Axes ---
figure('Position', [100, 100, 800, 650], 'Color', 'w');
ax = axes('FontSize', 11, 'FontName', 'Helvetica', 'Box', 'on', 'LineWidth', 1);
hold(ax, 'on');

% --- Color and Style Definitions ---
ps_color = '#2ECC71';   % Emerald Green
rps_color = '#F39C12';  % Orange
nr_color = '#E74C3C';   % Alizarin Red
sf_color = '#3498DB';   % Peter River Blue
line_width = 2.5;
marker_size = 8;

% --- Plot Data Curves (No Marker Edges) ---
plot(ax, ps_data(:,1), ps_data(:,2), 'd-', 'Color', ps_color, 'MarkerFaceColor', ps_color, 'MarkerEdgeColor', ps_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'PS (Proposed)');
plot(ax, rps_data(:,1), rps_data(:,2), '^-', 'Color', rps_color, 'MarkerFaceColor', rps_color, 'MarkerEdgeColor', rps_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'RPS (Random Phase)');
plot(ax, nr_data(:,1), nr_data(:,2), 'p-', 'Color', nr_color, 'MarkerFaceColor', nr_color, 'MarkerEdgeColor', 'k', 'LineWidth', line_width, 'MarkerSize', marker_size+2, 'DisplayName', 'NR (No RIS)');
plot(ax, sf_data(:,1), sf_data(:,2), 'o-', 'Color', sf_color, 'MarkerFaceColor', sf_color, 'MarkerEdgeColor', sf_color, 'LineWidth', line_width, 'MarkerSize', marker_size, 'DisplayName', 'SF (Straight Flight)');

% --- Axis Limits and Grid ---
xlim([2 16]); ylim([2 13]);
grid on;
ax.GridLineStyle = '--';
ax.GridAlpha = 0.3;
ax.Layer = 'bottom';

% --- Annotations ---
% Beta_C labels
text(ax, ps_data(1,1), ps_data(1,2) + 0.5, '$\beta_C = 1$', 'Interpreter', 'latex', 'FontSize', 12, 'HorizontalAlignment', 'center');
text(ax, ps_data(end,1), ps_data(end,2) - 0.5, '$\beta_C = 0$', 'Interpreter', 'latex', 'FontSize', 12, 'HorizontalAlignment', 'center');

% Increasing beta_C curved arrow
x_arc = [12, 10, 8]; y_arc = [10.2, 11.5, 11.2];
p = polyfit(x_arc, y_arc, 2);
x_fit = linspace(min(x_arc), max(x_arc), 100);
y_fit = polyval(p, x_fit);
plot(ax, x_fit, y_fit, 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
h_q = quiver(ax, x_fit(2), y_fit(2), x_fit(1)-x_fit(2), y_fit(1)-y_fit(2), 0, 'Color', 'k', 'MaxHeadSize', 0.5, 'LineWidth', 1.5, 'HandleVisibility', 'off');
text(ax, 10, 11.8, 'Increasing $\beta_C$', 'Interpreter', 'latex', 'FontSize', 12, 'BackgroundColor', [1 1 1 0.7], 'Margin', 1);

% Gain Annotations with custom arrows
% Gain from RIS
draw_custom_arrow(ax, nr_data(3,:), ps_data(3,:), nr_color, 'Gain from introduction of RIS', 'left');
% Gain from Trajectory Design
draw_custom_arrow(ax, sf_data(5,:), nr_data(6,:), sf_color, 'Gain from trajectory design', 'right');
% Gain from Phase Shift
draw_custom_arrow(ax, rps_data(6,:), ps_data(7,:), rps_color, 'Gain from phase shift', 'right');

% --- Final Setup ---
xlabel('Average Sensing SNR (dB)', 'FontWeight', 'bold', 'FontSize', 12);
ylabel("CU's Average Sum-Rate (bits/s/Hz)", 'FontWeight', 'bold', 'FontSize', 12);
title('Rate-SNR Regions for Different Schemes', 'FontWeight', 'bold', 'FontSize', 14);
lgd = legend('show', 'Location', 'SouthWest', 'FontSize', 11);
lgd.BoxFace.ColorType = 'truecoloralpha';
lgd.BoxFace.ColorData = uint8(255*[1 1 1 0.8]'); % White with 80% opacity
hold(ax, 'off');

%% --- Helper function for custom arrows ---
function draw_custom_arrow(ax, p_start, p_end, color, label_text, text_align)
    % Draws a dashed arrow with text between two points
    line([p_start(1), p_end(1)], [p_start(2), p_end(2)], 'Color', color, 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    
    % Add arrow heads
    quiver(ax, p_start(1), p_start(2), p_end(1)-p_start(1), p_end(2)-p_start(2), 0.1, ...
        'Color', color, 'MaxHeadSize', 0.8, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    
    % Place text in the middle
    mid_point = (p_start + p_end) / 2;
    if strcmpi(text_align, 'left')
        text(mid_point(1) - 0.2, mid_point(2), label_text, 'Color', color, 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
    else
        text(mid_point(1) + 0.2, mid_point(2), label_text, 'Color', color, 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    end
end