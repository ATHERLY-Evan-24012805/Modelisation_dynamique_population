import numpy as np
from scipy.optimize import fsolve

def f(y, a):
    return y * a

def verhulst(y, a, K):
    return y + (a * y * (1 - (y / K)))

def lotka_voltera_x(a, b, x, y):
    return a * x - b * x * y

def lotka_voltera_y(c, d, x, y):
    return -c * y + d * x * y

def explicit_euler_method(y0, t0, h, t_max, a):
    t_values = np.arange(t0, t_max + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(len(t_values) - 1):
        y_values[i + 1] = y_values[i] + h * f(y_values[i], a)

    return t_values, y_values

def implicit_euler_method(y0, t0, h, t_max, a):
    t_values = np.arange(t0, t_max + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(len(t_values) - 1):
        # équation à résoudre: y_values[i + 1] = y_values[i] + h * f(y_values[i+1], a)
        # y_values[i + 1] - y_values[i] - h * f(y_values[i], a) = 0
        # Z - y_values[i] - h * f(Z) = 0
        fonction_eq = lambda Z : Z - y_values[i] - h * f(Z, a)
        # On cherche la meilleure estimation de y[i + 1]; y[i] est la meilleure car le pas est faible et le valeurs ne change pas beaucoup
        point_initial = y_values[i] # fsolve résout par itération on définit ici le point de départ
        y_values[i + 1] = fsolve(fonction_eq, point_initial)
    return t_values, y_values

def trapezes(y0, t0, h, t_max, a):
    t_values = np.arange(t0, t_max + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(len(t_values) - 1):
        # équation à résoudre: y_values[i+1] = y_values[i] + (h/2) * (f(y_values[i], a) + f(y_values[i+1], a))
        fonction_eq = lambda Z : Z - y_values[i] - (h / 2) * (f(y_values[i], a) + f(Z, a))
        point_initial = y_values[i]
        y_values[i + 1] = fsolve(fonction_eq, point_initial)
    return t_values, y_values

def trapezes2(y0, t0, h, t_max, a):
    t_values = np.arange(t0, t_max + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    for i in range(len(t_values) - 1):
        euler_explicit = y_values[i] + h * f(y_values[i], a)

        fonction_eq = lambda Z : Z - y_values[i] - h * f(Z, a)
        # On cherche la meilleure estimation de y[i + 1]; y[i] est la meilleure car le pas est faible et le valeurs ne change pas beaucoup
        point_initial = y_values[i] # fsolve résout par itération on définit ici le point de départ
        euler_implicite = fsolve(fonction_eq, point_initial)

        y_values[i+1] = (euler_explicit + euler_implicite)/2
    return t_values, y_values

def explicit_euler_method_lotka(y0, x0, t0, h, t_max, a, b, c, d):
    t_values = np.arange(t0, t_max + h, h)
    y_values = np.zeros(len(t_values))
    x_values = np.zeros(len(t_values))
    y_values[0] = y0
    x_values[0] = x0

    for i in range(len(t_values) - 1):
        x_temp = x_values[i]
        y_temp = y_values[i]
        y_values[i + 1] = y_values[i] + h * lotka_voltera_y(c, d, x_temp, y_temp)
        x_values[i + 1] = x_values[i] + h * lotka_voltera_x(a, b, x_temp, y_temp)

    return t_values, y_values, x_values