import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, BarycentricInterpolator
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class NewtonInterpolator:
    """
    Interpolador de Newton usando diferencias divididas
    """

    def __init__(self, x_points=None, y_points=None):
        """
        Inicializa el interpolador

        Args:
            x_points: array de coordenadas x (opcional)
            y_points: array de coordenadas y (opcional)
        """
        if x_points is not None and y_points is not None:
            self.set_points(x_points, y_points)
        else:
            self.x_points = np.array([])
            self.y_points = np.array([])
            self.n = 0
            self.divided_differences = None

    def set_points(self, x_points, y_points):
        """
        Establece los puntos de interpolación
        """
        self.x_points = np.array(x_points, dtype=float)
        self.y_points = np.array(y_points, dtype=float)
        self.n = len(x_points)

        if len(x_points) != len(y_points):
            raise ValueError("x_points e y_points deben tener la misma longitud")

        if len(np.unique(x_points)) != len(x_points):
            raise ValueError("Los puntos x deben ser únicos")

        # Calcular tabla de diferencias divididas
        self.divided_differences = self._compute_divided_differences()

    def _compute_divided_differences(self):
        """
        Calcula la tabla de diferencias divididas

        Returns:
            Matriz triangular superior con las diferencias divididas
        """
        # Crear tabla de diferencias divididas
        table = np.zeros((self.n, self.n))

        # Primera columna: f[x_i] = y_i
        table[:, 0] = self.y_points

        # Llenar la tabla usando la fórmula recursiva
        for j in range(1, self.n):
            for i in range(self.n - j):
                numerator = table[i + 1, j - 1] - table[i, j - 1]
                denominator = self.x_points[i + j] - self.x_points[i]
                table[i, j] = numerator / denominator

        return table

    def get_divided_differences_table(self):
        """
        Devuelve la tabla de diferencias divididas en formato legible
        """
        if self.divided_differences is None:
            return "No hay puntos definidos"

        # Crear DataFrame para mejor visualización
        columns = [f"f[x{i}]" if i == 0 else f"f[x0,...,x{i}]" for i in range(self.n)]
        df = pd.DataFrame(self.divided_differences, columns=columns)

        # Reemplazar ceros con NaN para mejor visualización
        for i in range(self.n):
            for j in range(i + 1, self.n):
                df.iloc[i, j] = np.nan

        return df

    def get_coefficients(self):
        """
        Obtiene los coeficientes del polinomio de Newton
        (primera fila de la tabla de diferencias divididas)
        """
        if self.divided_differences is None:
            return np.array([])

        return self.divided_differences[0, :]

    def interpolate_point(self, x):
        """
        Interpola un solo punto usando la forma de Newton
        P(x) = f[x0] + f[x0,x1](x-x0) + f[x0,x1,x2](x-x0)(x-x1) + ...
        """
        if self.n == 0:
            raise ValueError("No hay puntos definidos")

        result = self.divided_differences[0, 0]  # f[x0]
        product = 1.0

        for i in range(1, self.n):
            product *= (x - self.x_points[i - 1])
            result += self.divided_differences[0, i] * product

        return result

    def interpolate(self, x_eval):
        """
        Interpola múltiples puntos
        """
        x_eval = np.asarray(x_eval)

        if x_eval.ndim == 0:
            return self.interpolate_point(float(x_eval))

        return np.array([self.interpolate_point(x) for x in x_eval])

    def add_point(self, x_new, y_new):
        """
        Agrega un nuevo punto de forma incremental (muy eficiente)
        """
        # Verificar que el punto x no exista ya
        if x_new in self.x_points:
            raise ValueError(f"El punto x = {x_new} ya existe")

        # Agregar nuevo punto
        self.x_points = np.append(self.x_points, x_new)
        self.y_points = np.append(self.y_points, y_new)
        self.n += 1

        # Actualizar tabla de diferencias divididas incrementalmente
        if self.divided_differences is None:
            self.divided_differences = np.array([[y_new]])
        else:
            # Expandir tabla
            old_table = self.divided_differences
            new_table = np.zeros((self.n, self.n))
            new_table[:-1, :-1] = old_table

            # Nueva fila (diferencias de orden 0)
            new_table[-1, 0] = y_new

            # Calcular nuevas diferencias divididas
            for j in range(1, self.n):
                if self.n - j > 0:  # Hay suficientes puntos
                    i = self.n - j - 1
                    numerator = new_table[i + 1, j - 1] - new_table[i, j - 1]
                    denominator = self.x_points[i + j] - self.x_points[i]
                    new_table[i, j] = numerator / denominator

            self.divided_differences = new_table

    def get_newton_form_string(self):
        """
        Devuelve el polinomio en forma de Newton como string
        """
        if self.n == 0:
            return "P(x) = 0"

        coeffs = self.get_coefficients()
        terms = []

        # Primer término: f[x0]
        if abs(coeffs[0]) > 1e-10:
            terms.append(f"{coeffs[0]:.6g}")

        # Términos restantes
        for i in range(1, self.n):
            if abs(coeffs[i]) < 1e-10:
                continue

            # Construir el producto (x - x0)(x - x1)...(x - x_{i-1})
            product_terms = []
            for j in range(i):
                if abs(self.x_points[j]) < 1e-10:  # x_j ≈ 0
                    product_terms.append("x")
                elif self.x_points[j] > 0:
                    product_terms.append(f"(x - {self.x_points[j]:.6g})")
                else:
                    product_terms.append(f"(x + {abs(self.x_points[j]):.6g})")

            product_str = "".join(product_terms)

            # Formatear coeficiente
            if abs(coeffs[i] - 1) < 1e-10:
                coeff_str = ""
            elif abs(coeffs[i] + 1) < 1e-10:
                coeff_str = "-"
            else:
                coeff_str = f"{coeffs[i]:.6g}"

            term = f"{coeff_str}{product_str}" if coeff_str else product_str
            terms.append(term)

        if not terms:
            return "P(x) = 0"

        # Unir términos
        result = "P(x) = " + terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"

        return result

    def to_standard_form(self):
        """
        Convierte el polinomio de Newton a forma estándar
        usando expansión simbólica
        """
        if self.n == 0:
            return np.array([0])

        # Empezar con el polinomio P(x) = f[x0]
        poly = np.array([self.divided_differences[0, 0]])

        # Agregar cada término de la forma de Newton
        for i in range(1, self.n):
            coeff = self.divided_differences[0, i]

            if abs(coeff) < 1e-14:
                continue

            # Construir el polinomio (x - x0)(x - x1)...(x - x_{i-1})
            product_poly = np.array([1])  # Polinomio constante 1

            for j in range(i):
                # Multiplicar por (x - x_j)
                factor = np.array([1, -self.x_points[j]])
                product_poly = np.convolve(product_poly, factor)

            # Multiplicar por el coeficiente
            product_poly *= coeff

            # Sumar al polinomio total
            # Ajustar tamaños para la suma
            max_len = max(len(poly), len(product_poly))
            poly_padded = np.pad(poly, (max_len - len(poly), 0))
            product_padded = np.pad(product_poly, (max_len - len(product_poly), 0))

            poly = poly_padded + product_padded

        return poly

    def get_standard_form_string(self):
        """
        Devuelve el polinomio en forma estándar como string
        """
        coeffs = self.to_standard_form()
        if len(coeffs) == 0 or (len(coeffs) == 1 and abs(coeffs[0]) < 1e-10):
            return "P(x) = 0"

        terms = []
        n = len(coeffs)

        for i, coeff in enumerate(coeffs):
            if abs(coeff) < 1e-10:
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
            return "P(x) = 0"

        # Unir términos
        result = "P(x) = " + terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"

        return result


class InterpolationComparator:
    """
    Clase para comparar Newton vs otros métodos
    """

    @staticmethod
    def compare_methods(x_data, y_data, x_eval):
        """
        Compara diferentes métodos de interpolación
        """
        methods = {}

        # Newton (diferencias divididas)
        newton_interp = NewtonInterpolator(x_data, y_data)
        methods['Newton'] = newton_interp.interpolate(x_eval)

        # Lagrange
        lagrange_poly = lagrange(x_data, y_data)
        methods['Lagrange'] = lagrange_poly(x_eval)

        # Barycentric (numéricamente estable)
        bary_interp = BarycentricInterpolator(x_data, y_data)
        methods['Barycentric'] = bary_interp(x_eval)

        return methods

    @staticmethod
    def efficiency_test(max_points=20):
        """
        Compara la eficiencia de agregar puntos incrementalmente
        """
        import time

        # Función de prueba
        def f(x):
            return np.sin(x) + 0.5 * np.cos(3 * x)

        x_base = np.linspace(0, 2 * np.pi, max_points)
        y_base = f(x_base)

        print("=== Test de Eficiencia: Agregar Puntos Incrementalmente ===")

        # Método incremental (Newton)
        newton_interp = NewtonInterpolator()
        start_time = time.time()

        for i in range(max_points):
            newton_interp.add_point(x_base[i], y_base[i])

        newton_time = time.time() - start_time

        # Método tradicional (recalcular todo cada vez)
        start_time = time.time()

        for i in range(1, max_points + 1):
            _ = NewtonInterpolator(x_base[:i], y_base[:i])

        traditional_time = time.time() - start_time

        print(f"Tiempo método incremental: {newton_time:.4f} s")
        print(f"Tiempo método tradicional: {traditional_time:.4f} s")
        print(f"Speedup: {traditional_time / newton_time:.2f}x")


def ejemplo_basico():
    """Ejemplo básico de interpolación de Newton"""
    print("=== Ejemplo Básico: Interpolación de Newton ===")

    # Puntos de datos
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([1, 4, 1, 3, 2])

    # Crear interpolador
    interp = NewtonInterpolator(x_data, y_data)

    # Mostrar tabla de diferencias divididas
    print("Tabla de diferencias divididas:")
    print(interp.get_divided_differences_table())
    print()

    # Mostrar polinomios
    print("Forma de Newton:")
    print(interp.get_newton_form_string())
    print()
    print("Forma estándar:")
    print(interp.get_standard_form_string())

    # Puntos para evaluar
    x_eval = np.linspace(-0.5, 4.5, 100)
    y_interp = interp.interpolate(x_eval)

    # Graficar
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_data, y_data, 'ro', markersize=8, label='Puntos de datos')
    plt.plot(x_eval, y_interp, 'b-', linewidth=2, label='Interpolación de Newton')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación de Newton - Ejemplo Básico')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Mostrar coeficientes
    coeffs = interp.get_coefficients()
    plt.subplot(1, 2, 2)
    plt.bar(range(len(coeffs)), coeffs, alpha=0.7)
    plt.xlabel('Orden de diferencia dividida')
    plt.ylabel('Valor del coeficiente')
    plt.title('Coeficientes de las Diferencias Divididas')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def ejemplo_construccion_incremental():
    """Ejemplo de construcción incremental del polinomio"""
    print("=== Ejemplo: Construcción Incremental ===")

    # Función conocida para interpolación
    def f(x):
        return x ** 3 - 2 * x ** 2 + x + 1

    x_points = [0, 1, 2, 3, 4]

    # Crear interpolador vacío
    interp = NewtonInterpolator()

    plt.figure(figsize=(15, 10))

    for i, x in enumerate(x_points):
        y = f(x)
        interp.add_point(x, y)

        print(f"\nDespués de agregar punto ({x}, {y}):")
        print(f"Grado del polinomio: {interp.n - 1}")
        print(f"Forma de Newton: {interp.get_newton_form_string()}")

        # Graficar evolución
        plt.subplot(2, 3, i + 1)

        x_eval = np.linspace(-0.5, 4.5, 100)
        y_true = f(x_eval)
        y_interp = interp.interpolate(x_eval)

        plt.plot(x_eval, y_true, 'k--', alpha=0.5, label='Función verdadera')
        plt.plot(interp.x_points, interp.y_points, 'ro', markersize=8,
                 label=f'Puntos ({interp.n})')
        plt.plot(x_eval, y_interp, 'b-', linewidth=2,
                 label=f'Newton (grado {interp.n - 1})')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Paso {i + 1}: {interp.n} puntos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-2, 10)

    plt.tight_layout()
    plt.show()

    print(f"\nTabla final de diferencias divididas:")
    print(interp.get_divided_differences_table())


def ejemplo_comparacion_metodos():
    """Comparación entre Newton, Lagrange y Barycentric"""
    print("=== Comparación: Newton vs Lagrange vs Barycentric ===")

    # Función de prueba: polinomio de Runge
    def runge_function(x):
        return 1 / (1 + 25 * x ** 2)

    # Puntos de datos
    n_points = 9
    x_data = np.linspace(-1, 1, n_points)
    y_data = runge_function(x_data)

    # Puntos de evaluación
    x_eval = np.linspace(-1, 1, 200)
    y_true = runge_function(x_eval)

    # Comparar métodos
    methods = InterpolationComparator.compare_methods(x_data, y_data, x_eval)

    # Calcular errores
    errors = {}
    for name, values in methods.items():
        errors[name] = np.abs(values - y_true)

    # Mostrar estadísticas
    print(f"Estadísticas de error (con {n_points} puntos):")
    for name, error_vals in errors.items():
        max_error = np.max(error_vals)
        rms_error = np.sqrt(np.mean(error_vals ** 2))
        print(f"{name:12s}: Error máx = {max_error:.2e}, Error RMS = {rms_error:.2e}")

    # Graficar comparación
    plt.figure(figsize=(15, 5))

    # Interpolaciones
    plt.subplot(1, 2, 1)
    plt.plot(x_eval, y_true, 'k--', linewidth=2, label='Función verdadera')
    plt.plot(x_data, y_data, 'ko', markersize=6, label='Puntos de datos')

    colors = ['red', 'blue', 'green']
    for i, (name, values) in enumerate(methods.items()):
        plt.plot(x_eval, values, color=colors[i], linewidth=2,
                 label=name, alpha=0.8)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparación de Métodos de Interpolación')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 1.2)

    # Errores
    plt.subplot(1, 2, 2)
    for i, (name, error_vals) in enumerate(errors.items()):
        plt.semilogy(x_eval, error_vals, color=colors[i], linewidth=2,
                     label=name, alpha=0.8)

    plt.xlabel('x')
    plt.ylabel('Error absoluto')
    plt.title('Comparación de Errores')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def ejemplo_diferencias_divididas_detallado():
    """Ejemplo detallado del cálculo de diferencias divididas"""
    print("=== Ejemplo Detallado: Cálculo de Diferencias Divididas ===")

    # Puntos simples para cálculo manual
    x_data = np.array([1, 2, 3, 4])
    y_data = np.array([2, 8, 18, 32])  # f(x) = 2x^2

    interp = NewtonInterpolator(x_data, y_data)

    print("Función: f(x) = 2x²")
    print("Puntos de datos:")
    for i in range(len(x_data)):
        print(f"  x{i} = {x_data[i]}, f(x{i}) = {y_data[i]}")

    print("\nCálculo paso a paso de diferencias divididas:")

    # Diferencias de orden 0
    print("\nOrden 0 (valores de la función):")
    for i in range(len(x_data)):
        print(f"  f[x{i}] = {y_data[i]}")

    # Diferencias de orden 1
    print("\nOrden 1:")
    for i in range(len(x_data) - 1):
        diff = (y_data[i + 1] - y_data[i]) / (x_data[i + 1] - x_data[i])
        print(f"  f[x{i},x{i + 1}] = (f[x{i + 1}] - f[x{i}])/(x{i + 1} - x{i}) = "
              f"({y_data[i + 1]} - {y_data[i]})/({x_data[i + 1]} - {x_data[i]}) = {diff}")

    # Diferencias de orden 2
    print("\nOrden 2:")
    table = interp.divided_differences
    for i in range(len(x_data) - 2):
        diff = table[i, 2]
        print(f"  f[x{i},x{i + 1},x{i + 2}] = {diff}")

    print(f"\nTabla completa de diferencias divididas:")
    print(interp.get_divided_differences_table())

    print(f"\nPolinomio resultante:")
    print(f"Forma de Newton: {interp.get_newton_form_string()}")
    print(f"Forma estándar: {interp.get_standard_form_string()}")

    # Verificar que es correcto
    x_test = 2.5
    y_newton = interp.interpolate_point(x_test)
    y_true = 2 * x_test ** 2
    print(f"\nVerificación en x = {x_test}:")
    print(f"Newton: P({x_test}) = {y_newton}")
    print(f"Verdadero: f({x_test}) = {y_true}")
    print(f"Error: {abs(y_newton - y_true):.2e}")


def ejemplo_ventajas_newton():
    """Ejemplo que muestra las ventajas del método de Newton"""
    print("=== Ventajas del Método de Newton ===")

    # 1. Eficiencia al agregar puntos
    print("1. Eficiencia al agregar puntos incrementalmente:")
    InterpolationComparator.efficiency_test(15)

    # 2. Detección de redundancia
    print("\n2. Detección de puntos redundantes:")

    # Puntos que forman una parábola exacta
    x_data = [0, 1, 2]
    y_data = [1, 2, 5]  # f(x) = x² + 1

    interp = NewtonInterpolator(x_data, y_data)
    print(f"Polinomio con 3 puntos: {interp.get_newton_form_string()}")

    # Agregar un cuarto punto que está en la misma parábola
    x_new, y_new = 3, 10  # 3² + 1 = 10
    interp.add_point(x_new, y_new)

    print(f"Después de agregar (3, 10): {interp.get_newton_form_string()}")
    print("Tabla de diferencias divididas:")
    print(interp.get_divided_differences_table())

    coeffs = interp.get_coefficients()
    print(f"\nCoeficiente de orden 3: {coeffs[3]:.2e}")
    print("(Prácticamente cero, indicando que el punto es redundante)")

    # 3. Estabilidad numérica
    print("\n3. Análisis de estabilidad numérica:")

    # Comparar con puntos muy cercanos
    x_close = [0, 0.001, 0.002, 0.003]
    y_close = [1, 1.001, 1.004, 1.009]

    interp_close = NewtonInterpolator(x_close, y_close)
    coeffs_close = interp_close.get_coefficients()

    print("Puntos muy cercanos:")
    for i, (x, y) in enumerate(zip(x_close, y_close)):
        print(f"  ({x}, {y})")

    print("Coeficientes de diferencias divididas:")
    for i, coeff in enumerate(coeffs_close):
        print(f"  Orden {i}: {coeff:.6g}")


if __name__ == "__main__":
    print("📊 Interpolación de Newton - Diferencias Divididas\n")

    # Ejecutar ejemplos
    ejemplo_basico()
    print("\n" + "=" * 60 + "\n")

    ejemplo_diferencias_divididas_detallado()
    print("\n" + "=" * 60 + "\n")

    ejemplo_construccion_incremental()
    print("\n" + "=" * 60 + "\n")

    ejemplo_comparacion_metodos()
    print("\n" + "=" * 60 + "\n")

    ejemplo_ventajas_newton()

    print("\n✅ ¡Ejemplos completados!")
    print("\n📚 Conceptos clave del método de Newton:")
    print("• Diferencias divididas: f[x₀,...,xₖ] = (f[x₁,...,xₖ] - f[x₀,...,xₖ₋₁])/(xₖ - x₀)")
    print("• Forma de Newton: P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...")
    print("• Construcción incremental eficiente")
    print("• Detección automática de redundancia")
    print("• Equivalencia matemática con Lagrange")