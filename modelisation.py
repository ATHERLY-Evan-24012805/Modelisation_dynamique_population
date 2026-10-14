import numpy as np
from scipy.optimize import fsolve

def f(y, a):
    return y * a

def explicit_euler_method(y0, t0, h, t_max, a):
    t_values = np.arange(t0, t_max + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(len(t_values) - 1):
        y_values[i + 1] = y_values[i] + h * f(y_values[i], a)

    return t_values, y_values

def implicit_euler_method():
    t_values = np.arange(t0, t_max + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(len(t_values) - 1):
        # équation à résoudre: y_values[i + 1] = y_values[i] + h * f(y_values[i+1], a)
        # y_values[i + 1] - y_values[i] - h * f(y_values[i], a)
        # Z - y_values[i] - h * f(Z) = 0
        fonction_eq = lambda Z : Z - y_values[i] - h * f(Z) = 0
        point_initial = y_values[i]
        y_values[i + 1] = fsolve(fonction_eq, point_initial)
    return t_values, y_values