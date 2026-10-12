import numpy as np

def f(y, a):
    return y * a

def explicit_euler_method(f, y0, t0, h, t_max, a):
    t_values = np.arange(t0, t_max + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(len(t_values) - 1):
        y_values[i + 1] = y_values[i] + h * f(y_values[i], a)

    return t_values, y_values