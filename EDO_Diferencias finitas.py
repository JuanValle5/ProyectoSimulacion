import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings

warnings.filterwarnings('ignore')


class EDOSolver:
    """
    Solucionador de Ecuaciones Diferenciales Ordinarias usando diferencias finitas
    Resuelve ecuaciones de la forma: a*y'' + b*y' + c*y = f(x)
    con condiciones de frontera: y(x0) = y0, y(xf) = yf
    """

    def __init__(self, x_range, n_points, coeff_a=1, coeff_b=0, coeff_c=0):
        """
        Inicializa el solucionador

        Args:
            x_range: tupla (x_inicial, x_final)
            n_points: número de puntos de discretización
            coeff_a, coeff_b, coeff_c: coeficientes de la ecuación
        """
        self.x0, self.xf = x_range
        self.n = n_points
        self.h = (self.xf - self.x0) / (self.n - 1)
        self.x = np.linspace(self.x0, self.xf, self.n)
        self.a = coeff_a
        self.b = coeff_b
        self.c = coeff_c

    def solve_bvp(self, f_func, y0, yf):
        """
        Resuelve problema de valores en la frontera

        Args:
            f_func: función del lado derecho f(x)
            y0: valor en x0
            yf: valor en xf
        """
        # Crear matriz de coeficientes
        main_diag = -2 * self.a / self.h ** 2 + self.c
        upper_diag = self.a / self.h ** 2 + self.b / (2 * self.h)
        lower_diag = self.a / self.h ** 2 - self.b / (2 * self.h)

        # Matriz tridiagonal
        A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1],
                  shape=(self.n - 2, self.n - 2), format='csr')

        # Vector del lado derecho
        f_vals = np.array([f_func(xi) for xi in self.x[1:-1]])

        # Aplicar condiciones de frontera
        f_vals[0] -= lower_diag * y0
        f_vals[-1] -= upper_diag * yf

        # Resolver sistema
        y_interior = spsolve(A, f_vals)

        # Construir solución completa
        y = np.zeros(self.n)
        y[0] = y0
        y[-1] = yf
        y[1:-1] = y_interior

        return self.x, y

    def solve_ivp(self, f_func, y0, dy0):
        """
        Resuelve problema de valor inicial usando método de diferencias finitas
        Convierte a problema de frontera usando shooting method
        """
        # Estimación inicial para y(xf)
        yf_guess = y0 + dy0 * (self.xf - self.x0)

        # Iteración para encontrar yf correcto
        for _ in range(10):
            x, y = self.solve_bvp(f_func, y0, yf_guess)

            # Calcular derivada en x0 usando diferencias finitas
            dy0_calc = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * self.h)

            # Ajustar yf_guess
            error = dy0_calc - dy0
            if abs(error) < 1e-6:
                break
            yf_guess -= error * (self.xf - self.x0) / 2

        return x, y


class EDPSolver:
    """
    Solucionador de Ecuaciones Diferenciales Parciales usando diferencias finitas
    """

    def __init__(self, x_range, t_range, nx, nt):
        """
        Inicializa el solucionador para EDP

        Args:
            x_range: tupla (x_inicial, x_final)
            t_range: tupla (t_inicial, t_final)
            nx: número de puntos en x
            nt: número de puntos en t
        """
        self.x0, self.xf = x_range
        self.t0, self.tf = t_range
        self.nx = nx
        self.nt = nt
        self.dx = (self.xf - self.x0) / (self.nx - 1)
        self.dt = (self.tf - self.t0) / (self.nt - 1)
        self.x = np.linspace(self.x0, self.xf, self.nx)
        self.t = np.linspace(self.t0, self.tf, self.nt)

    def solve_heat_equation(self, alpha, initial_condition, boundary_conditions):
        """
        Resuelve la ecuación del calor: ∂u/∂t = α ∂²u/∂x²

        Args:
            alpha: coeficiente de difusión térmica
            initial_condition: función u(x,0)
            boundary_conditions: tupla (u(x0,t), u(xf,t))
        """
        # Parámetro de estabilidad
        r = alpha * self.dt / self.dx ** 2
        if r > 0.5:
            print(f"Advertencia: r = {r:.3f} > 0.5. Esquema puede ser inestable")

        # Inicializar matriz de solución
        u = np.zeros((self.nt, self.nx))

        # Condición inicial
        for i in range(self.nx):
            u[0, i] = initial_condition(self.x[i])

        # Condiciones de frontera
        u_left, u_right = boundary_conditions
        for j in range(self.nt):
            u[j, 0] = u_left(self.t[j]) if callable(u_left) else u_left
            u[j, -1] = u_right(self.t[j]) if callable(u_right) else u_right

        # Esquema explícito de diferencias finitas
        for j in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                u[j + 1, i] = u[j, i] + r * (u[j, i + 1] - 2 * u[j, i] + u[j, i - 1])

        return self.x, self.t, u

    def solve_wave_equation(self, c, initial_u, initial_ut, boundary_conditions):
        """
        Resuelve la ecuación de onda: ∂²u/∂t² = c² ∂²u/∂x²

        Args:
            c: velocidad de la onda
            initial_u: condición inicial u(x,0)
            initial_ut: condición inicial ∂u/∂t(x,0)
            boundary_conditions: tupla (u(x0,t), u(xf,t))
        """
        # Parámetro de estabilidad
        r = c * self.dt / self.dx
        if r > 1:
            print(f"Advertencia: r = {r:.3f} > 1. Esquema puede ser inestable")

        # Inicializar matriz de solución
        u = np.zeros((self.nt, self.nx))

        # Condición inicial u(x,0)
        for i in range(self.nx):
            u[0, i] = initial_u(self.x[i])

        # Condición inicial ∂u/∂t(x,0) usando diferencias finitas
        for i in range(1, self.nx - 1):
            u[1, i] = u[0, i] + self.dt * initial_ut(self.x[i]) + \
                      0.5 * r ** 2 * (u[0, i + 1] - 2 * u[0, i] + u[0, i - 1])

        # Condiciones de frontera
        u_left, u_right = boundary_conditions
        for j in range(self.nt):
            u[j, 0] = u_left(self.t[j]) if callable(u_left) else u_left
            u[j, -1] = u_right(self.t[j]) if callable(u_right) else u_right

        # Esquema explícito de diferencias finitas
        for j in range(1, self.nt - 1):
            for i in range(1, self.nx - 1):
                u[j + 1, i] = 2 * u[j, i] - u[j - 1, i] + \
                              r ** 2 * (u[j, i + 1] - 2 * u[j, i] + u[j, i - 1])

        return self.x, self.t, u


def ejemplo_edo():
    """Ejemplo: y'' + y = 0, y(0) = 0, y(π) = 0"""
    print("=== Ejemplo EDO: y'' + y = 0 ===")

    # Configurar solucionador
    solver = EDOSolver(x_range=(0, np.pi), n_points=50, coeff_a=1, coeff_c=1)

    # Función del lado derecho (cero en este caso)
    def f(x):
        return 0

    # Resolver
    x, y = solver.solve_bvp(f, y0=0, yf=0)

    # Solución analítica: y = A*sin(x) + B*cos(x)
    # Con y(0) = 0 y y(π) = 0, la solución es y = 0
    y_exact = np.zeros_like(x)

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Diferencias finitas', linewidth=2)
    plt.plot(x, y_exact, 'r--', label='Solución exacta', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solución de y\'\' + y = 0')
    plt.legend()
    plt.grid(True)
    plt.show()


def ejemplo_edo_no_homogenea():
    """Ejemplo: y'' - 4y = x, y(0) = 0, y(1) = 0"""
    print("=== Ejemplo EDO no homogénea: y'' - 4y = x ===")

    solver = EDOSolver(x_range=(0, 1), n_points=50, coeff_a=1, coeff_c=-4)

    def f(x):
        return x

    x, y = solver.solve_bvp(f, y0=0, yf=0)

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Diferencias finitas', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solución de y\'\' - 4y = x')
    plt.legend()
    plt.grid(True)
    plt.show()


def ejemplo_ecuacion_calor():
    """Ejemplo: Ecuación del calor con condiciones específicas"""
    print("=== Ejemplo EDP: Ecuación del calor ===")

    # Configurar solucionador
    solver = EDPSolver(x_range=(0, 1), t_range=(0, 0.1), nx=50, nt=100)

    # Parámetros
    alpha = 0.01  # coeficiente de difusión

    # Condición inicial: pulso gaussiano
    def initial_temp(x):
        return np.exp(-100 * (x - 0.5) ** 2)

    # Condiciones de frontera: temperatura cero en los extremos
    boundary_conditions = (0, 0)

    # Resolver
    x, t, u = solver.solve_heat_equation(alpha, initial_temp, boundary_conditions)

    # Graficar evolución temporal
    plt.figure(figsize=(12, 8))

    # Gráfico 3D
    plt.subplot(2, 2, 1)
    X, T = np.meshgrid(x, t)
    plt.contourf(X, T, u, levels=20, cmap='hot')
    plt.colorbar(label='Temperatura')
    plt.xlabel('Posición x')
    plt.ylabel('Tiempo t')
    plt.title('Evolución de temperatura')

    # Perfiles de temperatura en diferentes tiempos
    plt.subplot(2, 2, 2)
    time_indices = [0, 25, 50, 75, 99]
    for idx in time_indices:
        plt.plot(x, u[idx, :], label=f't = {t[idx]:.3f}')
    plt.xlabel('Posición x')
    plt.ylabel('Temperatura')
    plt.title('Perfiles de temperatura')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def ejemplo_ecuacion_onda():
    """Ejemplo: Ecuación de onda"""
    print("=== Ejemplo EDP: Ecuación de onda ===")

    # Configurar solucionador
    solver = EDPSolver(x_range=(0, 1), t_range=(0, 2), nx=100, nt=200)

    # Velocidad de la onda
    c = 0.5

    # Condición inicial: forma de campana
    def initial_u(x):
        return np.exp(-50 * (x - 0.3) ** 2)

    def initial_ut(x):
        return np.zeros_like(x)  # velocidad inicial cero

    # Condiciones de frontera: extremos fijos
    boundary_conditions = (0, 0)

    # Resolver
    x, t, u = solver.solve_wave_equation(c, initial_u, initial_ut, boundary_conditions)

    # Graficar
    plt.figure(figsize=(12, 8))

    # Gráfico 3D
    plt.subplot(2, 2, 1)
    X, T = np.meshgrid(x, t)
    plt.contourf(X, T, u, levels=20, cmap='viridis')
    plt.colorbar(label='Amplitud')
    plt.xlabel('Posición x')
    plt.ylabel('Tiempo t')
    plt.title('Propagación de onda')

    # Instantáneas en diferentes tiempos
    plt.subplot(2, 2, 2)
    time_indices = [0, 50, 100, 150, 199]
    for idx in time_indices:
        plt.plot(x, u[idx, :], label=f't = {t[idx]:.3f}')
    plt.xlabel('Posición x')
    plt.ylabel('Amplitud')
    plt.title('Instantáneas de la onda')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Solucionador de Ecuaciones Diferenciales - Método de Diferencias Finitas\n")

    # Ejecutar ejemplos
    ejemplo_edo()
    ejemplo_edo_no_homogenea()
    ejemplo_ecuacion_calor()
    ejemplo_ecuacion_onda()

    print("\n¡Ejemplos completados!")
    print("\nPuedes usar las clases EDOSolver y EDPSolver para resolver tus propias ecuaciones:")
    print("- EDOSolver: para ecuaciones diferenciales ordinarias")
    print("- EDPSolver: para ecuaciones diferenciales parciales (calor, onda, etc.)")