import numpy as np
import matplotlib.pyplot as plt

# Definición de la función que representa la EDO con efecto Allee
def allee_equation(N, t, r, K, A):
    return r * N * (1 - N / K) * (N / A - 1)

# Método de Euler
def euler_method(f, N0, t, r, K, A):
    N = np.zeros(len(t))
    N[0] = N0
    h = t[1] - t[0]
    for i in range(1, len(t)):
        N[i] = N[i-1] + h * f(N[i-1], t[i-1], r, K, A)
        # Evitar valores negativos
        if N[i] < 0:
            N[i] = 0
    return N

# Método de Runge-Kutta de Cuarto Orden (RK4)
def rk4_method(f, N0, t, r, K, A):
    N = np.zeros(len(t))
    N[0] = N0
    h = t[1] - t[0]
    for i in range(1, len(t)):
        k1 = f(N[i-1], t[i-1], r, K, A)
        k2 = f(N[i-1] + 0.5 * h * k1, t[i-1] + 0.5 * h, r, K, A)
        k3 = f(N[i-1] + 0.5 * h * k2, t[i-1] + 0.5 * h, r, K, A)
        k4 = f(N[i-1] + h * k3, t[i-1] + h, r, K, A)
        N[i] = N[i-1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Evitar valores negativos
        if N[i] < 0:
            N[i] = 0
    return N

# Parámetros específicos
r = 0.5          # Tasa de crecimiento
K = 50              # Capacidad de carga
A = 40          # Parámetro del efecto Allee
N0 = 30           # Condición inicial
h = 0.1            # Paso temporal
t_max = 100      # Tiempo máximo

t = np.arange(0, t_max + h, h)

# Cálculo de las soluciones numéricas
N_euler = euler_method(allee_equation, N0, t, r, K, A)
N_rk4 = rk4_method(allee_equation, N0, t, r, K, A)

# Gráfica comparativa
plt.figure(figsize=(10, 6))
#plt.plot(t, N_euler, label='Euler', linestyle='--')
plt.plot(t, N_rk4, label='RK4', linestyle='-')
plt.xlabel('Tiempo')
plt.ylabel('Población N(t)')
plt.title(f'(N0={N0}, r={r}, K={K}, A={A})')
plt.legend()
plt.grid(True)
plt.show()
