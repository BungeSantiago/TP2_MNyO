import numpy as np
import matplotlib.pyplot as plt

# Definición de la función que representa la EDO logística
def logistic_equation(N, t, r, K):
    return r * N * (1 - N / K)

# Método de Euler
def euler_method(f, N0, t, r, K):
    N = np.zeros(len(t))
    N[0] = N0
    h = t[1] - t[0]
    for i in range(1, len(t)):
        N[i] = N[i-1] + h * f(N[i-1], t[i-1], r, K)
    return N

# Método de Runge-Kutta de Segundo Orden (RK2)
def rk2_method(f, N0, t, r, K):
    N = np.zeros(len(t))
    N[0] = N0
    h = t[1] - t[0]
    for i in range(1, len(t)):
        k1 = f(N[i-1], t[i-1], r, K)
        k2 = f(N[i-1] + h * k1, t[i-1] + h, r, K)
        N[i] = N[i-1] + (h / 2) * (k1 + k2)
    return N

# Método de Runge-Kutta de Cuarto Orden (RK4)
def rk4_method(f, N0, t, r, K):
    N = np.zeros(len(t))
    N[0] = N0
    h = t[1] - t[0]
    for i in range(1, len(t)):
        k1 = f(N[i-1], t[i-1], r, K)
        k2 = f(N[i-1] + 0.5 * h * k1, t[i-1] + 0.5 * h, r, K)
        k3 = f(N[i-1] + 0.5 * h * k2, t[i-1] + 0.5 * h, r, K)
        k4 = f(N[i-1] + h * k3, t[i-1] + h, r, K)
        N[i] = N[i-1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return N

# Solución analítica de la ecuación logística
def analytical_solution(t, N0, r, K):
    return K * N0 * np.exp(r * t) / (K + N0 * (np.exp(r * t) - 1))

# Parámetros específicos
r = 0.1          # Tasa de crecimiento
K = 15         # Capacidad de carga
N0 = 2          # Condición inicial
h = 2          # Paso temporal
t_max = 100       # Tiempo máximo

t = np.arange(0, t_max + h, h)

# Cálculo de las soluciones numéricas
N_euler = euler_method(logistic_equation, N0, t, r, K)
N_rk4 = rk4_method(logistic_equation, N0, t, r, K)

# Cálculo de la solución analítica
N_exact = analytical_solution(t, N0, r, K)

# Gráfica comparativa
plt.figure(figsize=(10, 6))
plt.plot(t, N_euler, label='Euler', linestyle='--')
plt.plot(t, N_rk4, label='RK4', linestyle=':')
plt.xlabel('Tiempo')
plt.ylabel('Población N(t)')
plt.title(f'(N0={N0}, r={r})')
plt.legend()
plt.grid(True)
plt.show()
print(analytical_solution(1, 2, 0.1, 15))

