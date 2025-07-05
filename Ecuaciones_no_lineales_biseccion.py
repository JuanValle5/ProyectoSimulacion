from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import numpy as np

#Funcion de ejemplo:
# f(x) = x^2 + 15cos(x) - 40
#f(x) = x^3 + 15sen(x) + 10

#x**2 + 15*np.cos(x) - 40
#x**3 + 15*np.sin(x) + 10
def f(x):
    return x**2 + 15*np.cos(x) - 40  # Definición de la función a graficar


x = np.linspace(-10, 10, 100)  # Rango de valores para x



plt.plot(x, f(x)) # x = rango de valores y f(x) = función a graficar
plt.grid()
plt.axhline(y=0, linewidth=2, color='black')  # Línea horizontal en y=0
plt.axvline(x=0, linewidth=2, color='black')  # Línea vertical en x=0
plt.show()

solucion = root_scalar(f, method='bisect', bracket=[5, 6])  # Método de bisección para encontrar la raíz, bracket = un intervalo cerrado que contenga la raiz
print(f"Método de biseccion: \n\
        - Raiz = {solucion.root}\n\
        - interaciones = {solucion.iterations}\n\
        - Evaluaciones = {solucion.function_calls}")  # Numero de evaluaciones