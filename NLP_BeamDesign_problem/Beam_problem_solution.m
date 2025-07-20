clc;
clear;

%% Constants
rho_w   = 76.5e3;   % Weight density (N/m^3)
sigma_0 = 220e6;    % Permissible stress (Pa)
delta_0 = 0.02;     % Maximum deflection (m)
E       = 207e9;    % Young's modulus (Pa)
L       = 1;        % Length (m)
P       = 1e5;      % Concentrated load (N)
p0      = 1e6;      % Distributed load (N/m)

%% Design Variable Bounds
x1_min = 0.04;  x1_max = 0.12;
x2_min = 0.06;  x2_max = 0.20;

%% Create Grid for Visualization
x1 = linspace(0, 0.25, 500);
x2 = linspace(0, 0.25, 500);
[X1, X2] = meshgrid(x1, x2);

%% Calculate Responses
% Maximum bending moment
M_max = (P * L / 4) + (p0 * L^2 / 8);

% Stress calculation (Pa)
sigma = (6 * M_max) ./ (X1 .* X2.^2);

% Moment of inertia (m^4)
I = (X1 .* X2.^3) / 12;

% Deflection calculation (m)
deflection = (5 * p0 * L^4) ./ (384 * E * I) + (P * L^3) ./ (48 * E * I);

geom = X1-X2;

%% Determine Feasible Region
stress_feasible    = sigma <= sigma_0;
deflection_feasible = deflection <= delta_0;
geom_feasible = geom <= 0;
feasible = stress_feasible & deflection_feasible & geom_feasible;

% Restrict the feasible region to only points within the design variable bounds
feasible = feasible & (X1 >= x1_min) & (X1 <= x1_max) & (X2 >= x2_min) & (X2 <= x2_max);

%% Objective Function (Weight)
W = rho_w * X1 .* X2;

%% Plot Setup
figure;
hold on;
grid on;
colormap('parula');  % Set colormap

% --- Plot Feasible Region as a Transparent Overlay ---
% Convert the logical feasible matrix to double. Set infeasible areas to NaN.
feasible_overlay = double(feasible);
feasible_overlay(~feasible) = NaN;  
h_feasible = pcolor(X1, X2, feasible_overlay);
set(h_feasible, 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName','Feasible Region'); 

% --- Plot Constraint Boundaries ---
% Stress Constraint (red contour)
contour(X1, X2, sigma, [sigma_0, sigma_0], 'r', 'LineWidth', 2, ...
    'DisplayName', 'Stress Constraint');
% Deflection Constraint (blue contour)
contour(X1, X2, deflection, [delta_0, delta_0], 'b', 'LineWidth', 2, ...
    'DisplayName', 'Deflection Constraint');

% Geometric Constraint (blue contour)
contour(X1, X2, geom, [0, 0], 'g', 'LineWidth', 2, ...
    'DisplayName', 'Geometric Constraint');

% --- Plot Design Variable Bounds ---
% Vertical lines for x1 bounds
plot([x1_min x1_min], ylim, 'Color', [0.2 0.5 0.3], 'LineWidth', 1.5, 'DisplayName', 'lb of x1');
plot([x1_max x1_max], ylim, 'Color', [0.7 0.3 0.4], 'LineWidth', 1.5, 'DisplayName', 'ub of x1');

% Horizontal lines for x2 bounds
plot(xlim, [x2_min x2_min], 'Color', [0.1 0.2 0.3], 'LineWidth', 1.5, 'DisplayName', 'lb of x2');
plot(xlim, [x2_max x2_max], 'Color', [0 0.3 0.6], 'LineWidth', 1.5, 'DisplayName', 'ub of x2');

% --- Plot Objective Function Contours ---
[C, h] = contour(X1, X2, W, 15, 'k--', 'ShowText', 'on', 'DisplayName', 'Weight of Beam');
clabel(C, h, 'FontSize', 8);

xlabel('x_1 (m)');
ylabel('x_2 (m)');
title('Graphical Optimization of Beam');

%% Optimization Verification using fmincon
objective = @(x) rho_w * x(1) * x(2);
lb = [x1_min, x2_min];
ub = [x1_max, x2_max];
x0 = [0.04, 0.06];
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
[x_opt, fval] = fmincon(objective, x0, [], [], [], [], lb, ub, @beam_constraints, options);

% Plot the optimal point
plot(x_opt(1), x_opt(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Optimal Point');

legend('Location', 'southeast');
%axis([0 0.16 0 0.26]);
hold off;

%% Nested Function for Constraints
function [c, ceq] = beam_constraints(x)
    % Local constants for constraint calculations
    E = 207e9;
    L = 1;
    P = 1e5;
    p0 = 1e6;
    sigma_0 = 220e6;
    delta_0 = 0.02;
    
    x1 = x(1);
    x2 = x(2);
    
    I = (x1 * x2^3) / 12;
    M_max = (P * L / 4) + (p0 * L^2 / 8);
    
    % Stress constraint: must be less than sigma_0
    sigma = (6 * M_max) / (x1 * x2^2);
    stress_constr = sigma - sigma_0;
    
    % Deflection constraint: must be less than delta_0
    deflection = (5 * p0 * L^4) / (384 * E * I) + (P * L^3) / (48 * E * I);
    deflection_constr = deflection - delta_0;

    geom = x1-x2;
    
    % Nonlinear inequality constraints: c(x) <= 0
    c = [stress_constr; deflection_constr; geom ];
    ceq = [];
end
