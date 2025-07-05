import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange as scipy_lagrange
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')


class LagrangeInterpolator:
    """
    Interpolador de Lagrange con múltiples métodos de implementación
    """

    def __init__(self, x_points, y_points):
        """
        Inicializa el interpolador con puntos de datos

        Args:
            x_points: array de coordenadas x
            y_points: array de coordenadas y
        """
        self.x_points = np.array(x_points, dtype=float)
        self.y_points = np.array(y_points, dtype=float)
        self.n = len(x_points)

        if len(x_points) != len(y_points):
            raise ValueError("x_points e y_points deben tener la misma longitud")

        if len(np.unique(x_points)) != len(x_points):
            raise ValueError("Los puntos x deben ser únicos")

    def lagrange_basis(self, x, i):
        """
        Calcula el i-ésimo polinomio base de Lagrange L_i(x)

        Args:
            x: punto donde evaluar
            i: índice del polinomio base

        Returns:
            Valor de L_i(x)
        """
        result = 1.0
        xi = self.x_points[i]

        for j in range(self.n):
            if i != j:
                xj = self.x_points[j]
                result *= (x - xj) / (xi - xj)

        return result

    def interpolate_point(self, x):
        """
        Interpola un solo punto usando la fórmula de Lagrange

        Args:
            x: punto a interpolar

        Returns:
            Valor interpolado P(x)
        """
        result = 0.0

        for i in range(self.n):
            result += self.y_points[i] * self.lagrange_basis(x, i)

        return result

    def interpolate(self, x_eval):
        """
        Interpola múltiples puntos

        Args:
            x_eval: array de puntos a evaluar

        Returns:
            Array de valores interpolados
        """
        x_eval = np.asarray(x_eval)

        if x_eval.ndim == 0:
            return self.interpolate_point(float(x_eval))

        return np.array([self.interpolate_point(x) for x in x_eval])

    def get_polynomial_coefficients(self):
        """
        Obtiene los coeficientes del polinomio interpolador
        en forma estándar: a_n*x^n + ... + a_1*x + a_0

        Returns:
            Array de coeficientes (del término de mayor grado al menor)
        """
        # Usar numpy para expandir el polinomio
        poly = np.poly1d([0])  # Polinomio cero

        for i in range(self.n):
            # Construir el i-ésimo término y_i * L_i(x)
            li_poly = np.poly1d([1])  # Polinomio identidad

            for j in range(self.n):
                if i != j:
                    # Multiplicar por (x - x_j) / (x_i - x_j)
                    factor_poly = np.poly1d([1, -self.x_points[j]])
                    li_poly *= factor_poly / (self.x_points[i] - self.x_points[j])

            # Sumar y_i * L_i(x) al polinomio total
            poly += self.y_points[i] * li_poly

        return poly.coefficients

    def get_polynomial_string(self):
        """
        Devuelve una representación en string del polinomio
        """
        coeffs = self.get_polynomial_coefficients()
        terms = []
        n = len(coeffs)

        for i, coeff in enumerate(coeffs):
            if abs(coeff) < 1e-10:  # Ignorar coeficientes muy pequeños
                continue

            power = n - 1 - i

            # Formatear coeficiente
            if abs(coeff - 1) < 1e-10 and power > 0:
                coeff_str = ""
            elif abs(coeff + 1) < 1e-10 and power > 0:
                coeff_str = "-"
            else:
                coeff_str = f"{coeff:.6g}"

            # Formatear término
            if power == 0:
                term = coeff_str if coeff_str else "1"
            elif power == 1:
                term = f"{coeff_str}x" if coeff_str else "x"
            else:
                term = f"{coeff_str}x^{power}" if coeff_str else f"x^{power}"

            terms.append(term)

        if not terms:
            return "0"

        # Unir términos con signos apropiados
        result = terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"

        return result

    def calculate_error_bound(self, x_eval, f_derivative_max):
        """
        Calcula una cota superior del error de interpolación

        Args:
            x_eval: puntos donde calcular el error
            f_derivative_max: valor máximo de |f^(n)(x)| en el intervalo

        Returns:
            Cota superior del error
        """
        x_eval = np.asarray(x_eval)
        omega = np.ones_like(x_eval)

        # Calcular ω(x) = ∏(x - x_i)
        for xi in self.x_points:
            omega *= (x_eval - xi)

        # Error bound: |f(x) - P(x)| ≤ |ω(x)| * M / (n+1)!
        factorial = np.math.factorial(self.n)
        error_bound = np.abs(omega) * f_derivative_max / factorial

        return error_bound


class InterpolationAnalyzer:
    """
    Clase para análisis comparativo de métodos de interpolación
    """

    @staticmethod
    def compare_methods(x_data, y_data, x_eval, true_function=None):
        """
        Compara diferentes métodos de interpolación
        """
        methods = {}

        # Lagrange (implementación propia)
        lagrange_interp = LagrangeInterpolator(x_data, y_data)
        methods['Lagrange (propia)'] = lagrange_interp.interpolate(x_eval)

        # Lagrange (scipy)
        scipy_poly = scipy_lagrange(x_data, y_data)
        methods['Lagrange (scipy)'] = scipy_poly(x_eval)

        # Interpolación lineal por tramos
        linear_interp = interp1d(x_data, y_data, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
        methods['Lineal por tramos'] = linear_interp(x_eval)

        # Interpolación cúbica (si hay suficientes puntos)
        if len(x_data) >= 4:
            cubic_interp = interp1d(x_data, y_data, kind='cubic',
                                    bounds_error=False, fill_value='extrapolate')
            methods['Cúbica por tramos'] = cubic_interp(x_eval)

        # Calcular errores si se conoce la función verdadera
        if true_function is not None:
            true_values = true_function(x_eval)
            errors = {}
            for name, values in methods.items():
                errors[name] = np.abs(values - true_values)
            return methods, errors

        return methods, None

    @staticmethod
    def plot_comparison(x_data, y_data, x_eval, methods, errors=None, true_function=None):
        """
        Grafica comparación de métodos
        """
        plt.figure(figsize=(15, 5))

        # Gráfico de interpolaciones
        plt.subplot(1, 2, 1)

        # Puntos de datos originales
        plt.plot(x_data, y_data, 'ko', markersize=8, label='Puntos de datos', zorder=5)

        # Función verdadera si está disponible
        if true_function is not None:
            plt.plot(x_eval, true_function(x_eval), 'k--',
                     linewidth=2, label='Función verdadera', alpha=0.7)

        # Métodos de interpolación
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (name, values) in enumerate(methods.items()):
            plt.plot(x_eval, values, color=colors[i % len(colors)],
                     linewidth=2, label=name, alpha=0.8)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Comparación de Métodos de Interpolación')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Gráfico de errores
        if errors is not None:
            plt.subplot(1, 2, 2)
            for i, (name, error_vals) in enumerate(errors.items()):
                plt.semilogy(x_eval, error_vals, color=colors[i % len(colors)],
                             linewidth=2, label=name, alpha=0.8)

            plt.xlabel('x')
            plt.ylabel('Error absoluto')
            plt.title('Error de Interpolación')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def ejemplo_basico():
    """Ejemplo básico de interpolación de Lagrange"""
    print("=== Ejemplo Básico: Interpolación de Lagrange ===")

    # Puntos de datos
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([1, 3, 2, 5, 4])

    # Crear interpolador
    interp = LagrangeInterpolator(x_data, y_data)

    # Puntos para evaluar
    x_eval = np.linspace(-0.5, 4.5, 100)
    y_interp = interp.interpolate(x_eval)

    # Mostrar polinomio
    print(f"Polinomio interpolador:")
    print(f"P(x) = {interp.get_polynomial_string()}")

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ro', markersize=8, label='Puntos de datos')
    plt.plot(x_eval, y_interp, 'b-', linewidth=2, label='Interpolación de Lagrange')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación de Lagrange - Ejemplo Básico')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def ejemplo_funcion_conocida():
    """Ejemplo con función conocida para analizar error"""
    print("=== Ejemplo: Interpolación de sin(x) ===")

    # Función verdadera
    def f_true(x):
        return np.sin(x)

    # Puntos de datos
    x_data = np.linspace(0, 2 * np.pi, 6)
    y_data = f_true(x_data)

    # Crear interpolador
    interp = LagrangeInterpolator(x_data, y_data)

    # Puntos para evaluar
    x_eval = np.linspace(0, 2 * np.pi, 200)

    # Comparar métodos
    methods, errors = InterpolationAnalyzer.compare_methods(
        x_data, y_data, x_eval, f_true)

    # Mostrar estadísticas de error
    print(f"\nPolinomio de Lagrange (grado {len(x_data) - 1}):")
    print(f"P(x) = {interp.get_polynomial_string()}")

    print(f"\nEstadísticas de error:")
    for name, error_vals in errors.items():
        print(f"{name:20s}: Error máx = {np.max(error_vals):.2e}, "
              f"Error RMS = {np.sqrt(np.mean(error_vals ** 2)):.2e}")

    # Graficar comparación
    InterpolationAnalyzer.plot_comparison(x_data, y_data, x_eval, methods, errors, f_true)


def ejemplo_fenomeno_runge():
    """Ejemplo del fenómeno de Runge con puntos equidistantes"""
    print("=== Ejemplo: Fenómeno de Runge ===")

    # Función de Runge
    def runge_function(x):
        return 1 / (1 + 25 * x ** 2)

    # Diferentes números de puntos
    n_points_list = [6, 11, 16]
    x_eval = np.linspace(-1, 1, 200)

    plt.figure(figsize=(15, 5))

    for i, n_points in enumerate(n_points_list):
        plt.subplot(1, 3, i + 1)

        # Puntos equidistantes
        x_data = np.linspace(-1, 1, n_points)
        y_data = runge_function(x_data)

        # Interpolación de Lagrange
        interp = LagrangeInterpolator(x_data, y_data)
        y_interp = interp.interpolate(x_eval)

        # Graficar
        plt.plot(x_eval, runge_function(x_eval), 'k--', linewidth=2,
                 label='Función verdadera')
        plt.plot(x_data, y_data, 'ro', markersize=6, label='Puntos de datos')
        plt.plot(x_eval, y_interp, 'b-', linewidth=2,
                 label=f'Lagrange (n={n_points})')

        plt.ylim(-2, 2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Fenómeno de Runge - {n_points} puntos')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Observación: Con puntos equidistantes, aumentar el grado puede")
    print("empeorar la interpolación cerca de los bordes (fenómeno de Runge)")


def ejemplo_puntos_chebyshev():
    """Ejemplo usando puntos de Chebyshev para evitar el fenómeno de Runge"""
    print("=== Ejemplo: Puntos de Chebyshev vs Equidistantes ===")

    def runge_function(x):
        return 1 / (1 + 25 * x ** 2)

    n_points = 11
    x_eval = np.linspace(-1, 1, 200)

    # Puntos equidistantes
    x_equi = np.linspace(-1, 1, n_points)
    y_equi = runge_function(x_equi)

    # Puntos de Chebyshev
    k = np.arange(1, n_points + 1)
    x_cheb = np.cos((2 * k - 1) * np.pi / (2 * n_points))
    y_cheb = runge_function(x_cheb)

    # Interpolaciones
    interp_equi = LagrangeInterpolator(x_equi, y_equi)
    interp_cheb = LagrangeInterpolator(x_cheb, y_cheb)

    y_interp_equi = interp_equi.interpolate(x_eval)
    y_interp_cheb = interp_cheb.interpolate(x_eval)

    # Calcular errores
    y_true = runge_function(x_eval)
    error_equi = np.abs(y_interp_equi - y_true)
    error_cheb = np.abs(y_interp_cheb - y_true)

    # Graficar
    plt.figure(figsize=(15, 5))

    # Interpolaciones
    plt.subplot(1, 2, 1)
    plt.plot(x_eval, y_true, 'k--', linewidth=2, label='Función verdadera')
    plt.plot(x_equi, y_equi, 'ro', markersize=6, label='Puntos equidistantes')
    plt.plot(x_cheb, y_cheb, 'bs', markersize=6, label='Puntos de Chebyshev')
    plt.plot(x_eval, y_interp_equi, 'r-', linewidth=2, label='Lagrange (equidistantes)')
    plt.plot(x_eval, y_interp_cheb, 'b-', linewidth=2, label='Lagrange (Chebyshev)')
    plt.ylim(-1, 1.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación: Equidistantes vs Chebyshev')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Errores
    plt.subplot(1, 2, 2)
    plt.semilogy(x_eval, error_equi, 'r-', linewidth=2, label='Error equidistantes')
    plt.semilogy(x_eval, error_cheb, 'b-', linewidth=2, label='Error Chebyshev')
    plt.xlabel('x')
    plt.ylabel('Error absoluto')
    plt.title('Comparación de Errores')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Error máximo con puntos equidistantes: {np.max(error_equi):.2e}")
    print(f"Error máximo con puntos de Chebyshev: {np.max(error_cheb):.2e}")
    print(f"Mejora: {np.max(error_equi) / np.max(error_cheb):.1f}x")


def ejemplo_interactivo():
    """Permite al usuario ingresar sus propios puntos"""
    print("=== Ejemplo Interactivo ===")
    print("Ingresa puntos para interpolar (termina con punto vacío)")

    x_points = []
    y_points = []

    while True:
        try:
            x = input(f"x{len(x_points) + 1} (o Enter para terminar): ")
            if not x.strip():
                break
            x = float(x)

            y = input(f"y{len(y_points) + 1}: ")
            y = float(y)

            x_points.append(x)
            y_points.append(y)

        except ValueError:
            print("Por favor, ingresa números válidos")
        except KeyboardInterrupt:
            print("\nOperación cancelada")
            return

    if len(x_points) < 2:
        print("Se necesitan al menos 2 puntos para interpolar")
        return

    try:
        # Crear interpolador
        interp = LagrangeInterpolator(x_points, y_points)

        print(f"\nPolinomio interpolador de grado {len(x_points) - 1}:")
        print(f"P(x) = {interp.get_polynomial_string()}")

        # Evaluar en un punto específico
        while True:
            try:
                x_eval = input("\n¿En qué punto quieres evaluar? (Enter para salir): ")
                if not x_eval.strip():
                    break
                x_eval = float(x_eval)
                result = interp.interpolate_point(x_eval)
                print(f"P({x_eval}) = {result}")
            except ValueError:
                print("Por favor, ingresa un número válido")
            except KeyboardInterrupt:
                break

        # Graficar
        x_range = np.array(x_points)
        margin = (x_range.max() - x_range.min()) * 0.2
        x_plot = np.linspace(x_range.min() - margin, x_range.max() + margin, 200)
        y_plot = interp.interpolate(x_plot)

        plt.figure(figsize=(10, 6))
        plt.plot(x_points, y_points, 'ro', markersize=8, label='Puntos de datos')
        plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Interpolación de Lagrange')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Tu Interpolación de Lagrange')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("🔢 Interpolación de Lagrange - Proyecto Completo\n")

    # Ejecutar ejemplos
    ejemplo_basico()
    print("\n" + "=" * 50 + "\n")

    ejemplo_funcion_conocida()
    print("\n" + "=" * 50 + "\n")

    ejemplo_fenomeno_runge()
    print("\n" + "=" * 50 + "\n")

    ejemplo_puntos_chebyshev()
    print("\n" + "=" * 50 + "\n")

    # Ejemplo interactivo (comentado para ejecución automática)
    # ejemplo_interactivo()

    print("✅ ¡Ejemplos completados!")
    print("\n📚 Conceptos clave demostrados:")
    print("• Interpolación básica de Lagrange")
    print("• Comparación con otros métodos")
    print("• Fenómeno de Runge")
    print("• Uso de puntos de Chebyshev")
    print("• Análisis de errores")