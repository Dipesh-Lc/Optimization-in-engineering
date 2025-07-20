
clc;
clear;

%% Define Constants
rho       = 7850;       % Density (kg/m³)
l         = 5;          % Column length (m)
E         = 200e9;      % Young's modulus (Pa)
M         = 1000;       % Mass of tank (kg)
g         = 9.81;       % Gravity (m/s²)
sigma_max = 250e6;      % Permissible stress (Pa)
alpha_val = 0;

%% Define Optimization Settings
lb = [0.04, 0.04];      % Lower bounds for [x1, x2]
ub = [0.5, 0.5];        % Upper bounds for [x1, x2]
x0 = [0.1, 0.1];        % Initial guess

% fmincon options
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');

% Nonlinear constraints(<=0):
constr = @(x) deal( [ (M*g)/(x(1)*x(2)) - sigma_max; ...
    (M*g)/(x(1)*x(2)) - (pi^2*E*x(2)^2)/(48*l^2) ], [] );

%% (1) Minimize Weight Only
% Weight function: W = rho * l * x1 * x2
objective_weight = @(x) rho * l * x(1) * x(2);

[x_w, fval_w] = fmincon(objective_weight, x0, [], [], [], [], lb, ub, constr, options);
W_opt = rho * l * x_w(1) * x_w(2);
% Frequency at the weight-optimal design (using the chosen frequency formula)
f_at_weight = sqrt((E*x_w(1)*x_w(2)^3) / (4*l^3*(M + (33/140)*rho*l*x_w(1)*x_w(2))))/(2*pi);

fprintf('--- Minimize Weight Only ---\n');
fprintf('x1 = %.6f, x2 = %.6f\n', x_w(1), x_w(2));
fprintf('Weight = %.2f kg, Frequency = %.2f Hz\n\n', W_opt, f_at_weight);

%% (2) Maximize Frequency Only
% To maximize frequency, we minimize the negative frequency.
objective_freq = @(x) -sqrt((E*x(1)*x(2)^3) / (4*l^3*(M + (33/140)*rho*l*x(1)*x(2))))/(2*pi);

[x_f, fval_f] = fmincon(objective_freq, x0, [], [], [], [], lb, ub, constr, options);
f_max = sqrt((E*x_f(1)*x_f(2)^3) / (4*l^3*(M + (33/140)*rho*l*x_f(1)*x_f(2))))/(2*pi);
W_at_f = rho * l * x_f(1) * x_f(2);

fprintf('--- Maximize Frequency Only ---\n');
fprintf('x1 = %.6f, x2 = %.6f\n', x_f(1), x_f(2));
fprintf('Weight = %.2f kg, Frequency = %.2f Hz\n\n', W_at_f, f_max);

%% (3) Normalized Weighted Optimization

norm_obj = @(x) (alpha_val/W_opt)*(rho*l*x(1)*x(2)) - ((1-alpha_val)/f_max)*...
    (sqrt((E*x(1)*x(2)^3)/(4*l^3*(M+(33/140)*rho*l*x(1)*x(2))))/(2*pi));

[x_norm, fval_norm] = fmincon(norm_obj, x0, [], [], [], [], lb, ub, constr, options);

fprintf('--- Normalized Weighted Optimization (alpha = %.2f) ---\n', alpha_val);
fprintf('x1 = %.6f, x2 = %.6f\n', x_norm(1), x_norm(2));
fprintf('Weight = %.2f kg, Frequency = %.2f Hz\n\n', ...
    rho*l*x_norm(1)*x_norm(2), ...
    sqrt((E*x_norm(1)*x_norm(2)^3)/(4*l^3*(M+(33/140)*rho*l*x_norm(1)*x_norm(2))))/(2*pi));

%% Graphical Visualization of the Feasible Region & Objectives
% Create a grid over the design space (within bounds)
x1_vals = linspace(0, 0.8, 800);
x2_vals = linspace(0, 0.8, 800);
[X1, X2] = meshgrid(x1_vals, x2_vals);

% Evaluate the two constraint functions on the grid
c1 = (M*g) ./ (X1.*X2) - sigma_max;  % c1 <= 0 i.e, Direct Compressive Stress
c2 = (M*g) ./ (X1.*X2) - (pi^2 * E * X2.^2) ./ (48*l^2);  % c2 <= 0 i.e, Buckling Stress

% Define the overall feasible region (grid points that satisfy both constraints and the bounds)
feasible = (c1 <= 0) & (c2 <= 0)& ...
    (X1 >= lb(1)) & (X1 <= ub(1)) & (X2 >= lb(2)) & (X2 <= ub(2));

% Compute objective functions on the grid
% (a) Weight
Weight_grid = rho * l .* X1 .* X2;
% (b) Frequency (using our chosen formula)
Freq_grid = sqrt((E .* X1 .* X2.^3) ./ (4*l^3.*(M + (33/140)*rho*l.*X1.*X2)))/(2*pi);
% (c) Normalized Weighted Objective for the chosen α
NormObj_grid = (alpha_val/W_opt)*(rho*l.*X1.*X2) - ((1-alpha_val)/f_max)*...
    (sqrt((E .* X1 .* X2.^3)./(4*l^3.*(M+(33/140)*rho*l.*X1.*X2)))/(2*pi));

%% Plot 1: Feasible Region, Constraint Boundaries, and Weight Contours
figure;
subplot(1,2,1);
hold on;
grid on;
colormap('parula');

% Plot feasible region as a shaded (semi‐transparent) area.
feasible_overlay = double(feasible);
feasible_overlay(~feasible) = NaN;  % set infeasible points to NaN
h_feas = pcolor(X1, X2, feasible_overlay);
set(h_feas, 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName','Feasible Region');

% Plot constraint boundaries (where c1 = 0 and c2 = 0)
contour(X1, X2, c1, [0 0], 'r', 'LineWidth', 2, 'DisplayName', 'Stress Constraint');
contour(X1, X2, c2, [0 0], 'b', 'LineWidth', 2, 'DisplayName', 'Buckling Constraint');

% --- Plot Design Variable Bounds ---
% Vertical lines for d bounds:
plot([lb(1) lb(1)], ylim, 'Color', [0.2 0.5 0.3], 'LineWidth', 1.5, 'DisplayName', 'Lower bound x1');
plot([ub(1) ub(1)], ylim, 'Color', [0.7 0.3 0.4], 'LineWidth', 1.5, 'DisplayName', 'Upper bound x1');
% Horizontal lines for t bounds:
plot(xlim, [lb(2) lb(2)], 'Color', [0.1 0.2 0.3], 'LineWidth', 1.5, 'DisplayName', 'Lower bound x2');
plot(xlim, [ub(2) ub(2)], 'Color', [0 0.3 0.6], 'LineWidth', 1.5, 'DisplayName', 'Upper bound x2');

% Plot weight objective contours
[Cw, hW] = contour(X1, X2, Weight_grid, 15, 'm--', 'ShowText', 'on', 'DisplayName', 'Weight');

% --- Plot the Optimal Point ---
plot(x_w(1), x_w(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Optimal Point');

xlabel('x_1');
ylabel('x_2');
title('Feasible Region & Weight Contours');
legend('Location','northeast');
hold off;

%% Plot 2: Feasible Region, Constraint Boundaries, and Frequency Contours
subplot(1,2,2);
hold on;
grid on;
colormap('parula');

% Shaded feasible region (reuse feasible_overlay)
h_feas3 = pcolor(X1, X2, feasible_overlay);
set(h_feas3, 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName','Feasible Region');

% Plot constraint boundaries (where c1 = 0 and c2 = 0)
contour(X1, X2, c1, [0 0], 'r', 'LineWidth', 2, 'DisplayName', 'Stress Constraint');
contour(X1, X2, c2, [0 0], 'b', 'LineWidth', 2, 'DisplayName', 'Buckling Constraint');


% --- Plot Design Variable Bounds ---
% Vertical lines for d bounds:
plot([lb(1) lb(1)], ylim, 'Color', [0.2 0.5 0.3], 'LineWidth', 1.5, 'DisplayName', 'Lower bound x1');
plot([ub(1) ub(1)], ylim, 'Color', [0.7 0.3 0.4], 'LineWidth', 1.5, 'DisplayName', 'Upper bound x1');
% Horizontal lines for t bounds:
plot(xlim, [lb(2) lb(2)], 'Color', [0.1 0.2 0.3], 'LineWidth', 1.5, 'DisplayName', 'Lower bound x2');
plot(xlim, [ub(2) ub(2)], 'Color', [0 0.3 0.6], 'LineWidth', 1.5, 'DisplayName', 'Upper bound x2');

% Plot frequency contours (using Freq_grid computed earlier)
[Cf, hF] = contour(X1, X2, Freq_grid, 15, 'c--', 'ShowText', 'on', 'DisplayName', 'Frequency');
clabel(Cf, hF, 'FontSize',8);

% --- Plot the Optimal Point ---
plot(x_f(1), x_f(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'DisplayName', 'Optimal Point');

xlabel('x_1');
ylabel('x_2');
title('Feasible Region & Frequency Contours');
legend('Location','northeast');
hold off;

%% Plot 3: Feasible Region & Normalized Weighted Objective for Multiple α values
alpha_vals = linspace(0, 0.03, 6);  % Change to linspace(0, 0.04, 8) for 8 subplots
num_alpha = length(alpha_vals);
rows = 2;  % Number of rows in subplot grid
cols = ceil(num_alpha / rows); % Number of columns in subplot grid

figure;
colormap('parula');

all_handles = []; % Collect legend handles
all_labels = {};  % Collect legend labels

for i = 1:num_alpha
    alpha_val = alpha_vals(i); % Set the current alpha
    
    % Recalculate the optimal x_norm for this α
    norm_obj = @(x) (alpha_val/W_opt)*(rho*l*x(1)*x(2)) - ((1-alpha_val)/f_max)*...
        (sqrt((E*x(1)*x(2)^3)/(4*l^3*(M+(33/140)*rho*l*x(1)*x(2)))))/(2*pi);
    [x_norm, ~] = fmincon(norm_obj, x0, [], [], [], [], lb, ub, constr, options);
    
    % Recalculate Normalized Weighted Objective for this alpha
    NormObj_grid = (alpha_val/W_opt)*(rho*l.*X1.*X2) - ((1-alpha_val)/f_max)*...
        (sqrt((E .* X1 .* X2.^3)./(4*l^3.*(M+(33/140)*rho*l.*X1.*X2)))/(2*pi));

    subplot(rows, cols, i);
    hold on;
    grid on;

    % Shaded feasible region
    h_feas = pcolor(X1, X2, feasible_overlay);
    set(h_feas, 'EdgeColor', 'none', 'FaceAlpha', 0.3);

    % Plot constraint boundaries
    h_c1 = contour(X1, X2, c1, [0 0], 'r', 'LineWidth', 2);
    h_c2 = contour(X1, X2, c2, [0 0], 'b', 'LineWidth', 2);

    % Design variable bounds
    h_lb_x1 = plot([lb(1) lb(1)], ylim, 'Color', [0.2 0.5 0.3], 'LineWidth', 1.5);
    h_ub_x1 = plot([ub(1) ub(1)], ylim, 'Color', [0.7 0.3 0.4], 'LineWidth', 1.5);
    h_lb_x2 = plot(xlim, [lb(2) lb(2)], 'Color', [0.1 0.2 0.3], 'LineWidth', 1.5);
    h_ub_x2 = plot(xlim, [ub(2) ub(2)], 'Color', [0 0.3 0.6], 'LineWidth', 1.5);

    % Contours of the normalized weighted objective
    [Cn, hn] = contour(X1, X2, NormObj_grid, 10, 'm--', 'ShowText','on');

    % Mark the optimal design found for this α
    h_opt = plot(x_norm(1), x_norm(2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor','r');

    xlabel('x_1');
    ylabel('x_2');
    title(['\alpha = ', num2str(alpha_val)]);
    hold off;
end
sgtitle('Normalized Weighted Optimization for Multiple \alpha Values');  % Super title for all subplots


%% Plot 4: Variation with α
% We now vary α (from 0 to 0.1) and record the optimal weight, frequency, and design variables.
N = 21;
alpha_vals = linspace(0, 0.1, N);
opt_weights = zeros(size(alpha_vals));
opt_freqs   = zeros(size(alpha_vals));
opt_x1      = zeros(size(alpha_vals));
opt_x2      = zeros(size(alpha_vals));

for i = 1:length(alpha_vals)
    a = alpha_vals(i);
    % Normalized weighted objective function for current α:
    norm_obj_i = @(x) (a/W_opt)*(rho*l*x(1)*x(2)) - ((1-a)/f_max)*...
        (sqrt((E*x(1)*x(2)^3)/(4*l^3*(M+(33/140)*rho*l*x(1)*x(2))))/(2*pi));
    [x_opt_i, ~] = fmincon(norm_obj_i, x0, [], [], [], [], lb, ub, constr, options);
    opt_x1(i) = x_opt_i(1);
    opt_x2(i) = x_opt_i(2);
    opt_weights(i) = rho*l*x_opt_i(1)*x_opt_i(2);
    opt_freqs(i) = sqrt((E*x_opt_i(1)*x_opt_i(2)^3)/(4*l^3*(M+(33/140)*rho*l*x_opt_i(1)*x_opt_i(2))))/(2*pi);
end

% Plot optimal weight vs. α
figure;
subplot(2,2,1);
plot(alpha_vals, opt_weights, '-o','LineWidth',1.5);
xlabel('\alpha'); ylabel('Optimal Weight (kg)');
title('Optimal Weight vs. \alpha'); grid on;

% Plot optimal frequency vs. α
subplot(2,2,2);
plot(alpha_vals, opt_freqs, '-o','LineWidth',1.5);
xlabel('\alpha'); ylabel('Optimal Frequency (Hz)');
title('Optimal Frequency vs. \alpha'); grid on;

% Plot optimal x1 vs. α
subplot(2,2,3);
plot(alpha_vals, opt_x1, '-o','LineWidth',1.5, 'DisplayName', 'x1');
hold on;
plot(alpha_vals, opt_x2, '-s','LineWidth',1.5, 'DisplayName', 'x2');
xlabel('\alpha','FontSize',12);
ylabel('Optimal values','FontSize',12);
title('Optimal x1 and x2 vs. \alpha','FontSize',14);
legend('show','Location','best');
grid on;
hold off;

% Plot optimal Weight vs. Frequency (trade-off curve)
subplot(2,2,4);
plot(opt_weights, opt_freqs, 'p-', 'MarkerSize',10, 'LineWidth',1.5);
xlabel('Optimal Weight (kg)'); ylabel('Optimal Frequency (Hz)');
title('Optimal Weight vs. Frequency'); grid on;

