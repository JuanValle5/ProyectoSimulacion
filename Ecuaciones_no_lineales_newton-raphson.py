from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import numpy as np

#Funcion de ejemplo:
# f(x) = x^2 + 15cos(x) - 40

def f(x):
    return x**2 + 15*np.cos(x) - 40  # Definición de la función a graficar

# Derivada de la función f(x) para el método de Newton-Raphson
def df(x):
    return 2*x - 15*np.sin(x)  # Derivada de la función f(x)


x = np.linspace(-10, 10, 100)  # Rango de valores para x



plt.plot(x, f(x)) # x = rango de valores y f(x) = función a graficar
plt.grid()
plt.axhline(y=0, linewidth=2, color='black')  # Línea horizontal en y=0
plt.axvline(x=0, linewidth=2, color='black')  # Línea vertical en x=0
plt.show()

solucion = root_scalar(f, method='Newton', x0 = 5, fprime= df)  # Método de Newton-raphson para encontrar la raíz. x0 = valor inicial, fprime = derivada de la función f(x)
print(f"Método de Newton-Raphson: \n\
        - Raiz = {solucion.root}\n\
        - interaciones = {solucion.iterations}\n\
        - Evaluaciones = {solucion.function_calls}")  # Imprime la raíz encontrada