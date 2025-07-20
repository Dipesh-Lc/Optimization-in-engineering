clc;
clear;

%% Constants
b = 0.5;
h = 1;
rho = 77000;
W = 25000;
E = 200e9;
sigma_y = 310e6;

%% Design Variable Bounds
% d and t must be positive.
lb = [0.02; 0.005];
ub = [0.05; 0.05];

%% Create Grid for Visualization
% We use a slightly larger range than the bounds for context.
d_vals = linspace(0, 0.06, 400);
t_vals = linspace(0, 0.06, 400);
[D, T] = meshgrid(d_vals, t_vals);

%% Compute Constraint Functions on the Grid
% Constraint 1: (W*sqrt(b^2+h^2))/(pi*h*sigma_y) - d*t <= 0
g1 = (W * sqrt(b^2 + h^2)) / (pi * h * sigma_y) - (D .* T);

% Constraint 2: (8*W*(b^2+h^2)^(1.5))/(pi^3*E*h) - (d^3*t + t^3*d) <= 0
g2 = (8 * W * (b^2 + h^2)^(1.5)) / (pi^3 * E * h) - (D.^3 .* T + T.^3 .* D);

% Constraint 3: t - d <= 0  (i.e. t <= d)
g3 = T - D;
%% Determine the Feasible Region
% The feasible region is where ALL constraints are satisfied and the design variables lie within bounds.
feasible = (g1 <= 0) & (g2 <= 0) & (g3 <= 0) & ...
    (D >= lb(1)) & (D <= ub(1)) & (T >= lb(2)) & (T <= ub(2));

%% Objective Function (Weight) over the Grid
F = D .* T* 2 * sqrt(b^2 + h^2) * rho * pi;
%% Plot Setup
figure;
hold on;
grid on;
colormap('parula');

% --- Plot the Feasible Region as a Transparent Overlay ---
% Convert the logical feasible matrix to double; set infeasible areas to NaN.
feasible_overlay = double(feasible);
feasible_overlay(~feasible) = NaN;
h_feasible = pcolor(D, T, feasible_overlay);
set(h_feasible, 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName','Feasible Region');

% --- Plot Constraint Boundaries ---
% g1 = 0 boundary (red)
[C1, h1] = contour(D, T, g1, [0 0], 'r', 'LineWidth', 2, 'DisplayName', 'Stress Constraints');
% g2 = 0 boundary (blue)
[C2, h2] = contour(D, T, g2, [0 0], 'b', 'LineWidth', 2, 'DisplayName', 'Buckling Constraints');
% g3 = 0 boundary (green)
[C3, h3] = contour(D, T, g3, [0 0], 'g', 'LineWidth', 2, 'DisplayName', 'Geometric Constriant: t<=d');

% --- Plot Design Variable Bounds ---
% Vertical lines for d bounds:
plot([lb(1) lb(1)], ylim, 'Color', [0.2 0.5 0.3], 'LineWidth', 1.5, 'DisplayName', 'Lower bound d');
plot([ub(1) ub(1)], ylim, 'Color', [0.7 0.3 0.4], 'LineWidth', 1.5, 'DisplayName', 'Upper bound d');
% Horizontal lines for t bounds:
plot(xlim, [lb(2) lb(2)], 'Color', [0.1 0.2 0.3], 'LineWidth', 1.5, 'DisplayName', 'Lower bound t');
plot(xlim, [ub(2) ub(2)], 'Color', [0 0.3 0.6], 'LineWidth', 1.5, 'DisplayName', 'Upper bound t');

% --- Plot Objective Function Contours ---
[C_obj, h_obj] = contour(D, T, F, 30, 'm--', 'ShowText', 'on', 'DisplayName', 'Weight of Truss');
clabel(C_obj, h_obj, 'FontSize', 8);

xlabel('d (m)');
ylabel('t (m)');
title('Graphical Optimization of Truss');

%% Solve the Problem Using fmincon
x0 = [0.01; 0.04];  % Initial guess
nonlcon = @truss_constraints;
options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'iter');
[x_opt, fval] = fmincon(@(x) x(1)*x(2), x0, [], [], [], [], lb, ub, nonlcon, options);

% Compute the scaled weight as given in the problem:
weight_total = fval * 2 * sqrt(b^2 + h^2) * rho * pi;
fprintf('Optimal Solution:\n');
fprintf('d = %.6f m\n', x_opt(1));
fprintf('t = %.6f m\n', x_opt(2));
fprintf('Minimum Weight = %.6f\n', weight_total);

% --- Plot the Optimal Point ---
plot(x_opt(1), x_opt(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Optimal Point');

legend('Location', 'northeastoutside');
xlim([0 0.06]);
ylim([0 0.06]);
hold off;
%% Nested Function for Constraints
function [c, ceq] = truss_constraints(x)
% Constants
b = 0.5;
h = 1;
W = 25000;
E = 200e9;
sigma_y = 310e6;

d = x(1);
t = x(2);

% Constraint 1: (W*sqrt(b^2+h^2))/(pi*h*sigma_y) - d*t <= 0
c1 = (W * sqrt(b^2+h^2)) / (pi * h * sigma_y) - d*t;

% Constraint 2: (8*W*(b^2+h^2)^(1.5))/(pi^3*E*h) - (d^3*t + t^3*d) <= 0
c2 = (8 * W * (b^2+h^2)^(1.5)) / (pi^3 * E * h) - (d^3*t + t^3*d);

% Constraint 3: t - d <= 0  (i.e. t <= d)
c3 = t - d;

c = [c1; c2; c3];
ceq = [];
end
