import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')


class MinimosCuadrados:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)

    def aproximacion_lineal(self):
        """
        Aproximación lineal: y = ax + b
        """
        # Cálculo de coeficientes usando las fórmulas de mínimos cuadrados
        sum_x = np.sum(self.x)
        sum_y = np.sum(self.y)
        sum_xy = np.sum(self.x * self.y)
        sum_x2 = np.sum(self.x ** 2)

        # Coeficientes a y b
        a = (self.n * sum_xy - sum_x * sum_y) / (self.n * sum_x2 - sum_x ** 2)
        b = (sum_y - a * sum_x) / self.n

        # Función resultante
        def f_linear(x):
            return a * x + b

        # Coeficiente de correlación
        r2 = self.coeficiente_correlacion(self.y, f_linear(self.x))

        return {
            'coeficientes': {'a': a, 'b': b},
            'funcion': f_linear,
            'ecuacion': f'y = {a:.4f}x + {b:.4f}',
            'r_cuadrado': r2
        }

    def aproximacion_polinomial(self, grado=2):
        """
        Aproximación polinómica: y = a_n*x^n + ... + a_1*x + a_0
        """
        # Usando numpy.polyfit para ajuste polinomial
        coeficientes = np.polyfit(self.x, self.y, grado)

        # Función polinomial
        def f_poly(x):
            return np.polyval(coeficientes, x)

        # Crear ecuación string
        ecuacion_partes = []
        for i, coef in enumerate(coeficientes):
            potencia = grado - i
            if potencia == 0:
                ecuacion_partes.append(f'{coef:.4f}')
            elif potencia == 1:
                ecuacion_partes.append(f'{coef:.4f}x')
            else:
                ecuacion_partes.append(f'{coef:.4f}x^{potencia}')

        ecuacion = 'y = ' + ' + '.join(ecuacion_partes).replace('+ -', '- ')

        # Coeficiente de correlación
        r2 = self.coeficiente_correlacion(self.y, f_poly(self.x))

        return {
            'coeficientes': coeficientes,
            'funcion': f_poly,
            'ecuacion': ecuacion,
            'r_cuadrado': r2,
            'grado': grado
        }

    def aproximacion_exponencial(self):
        """
        Aproximación exponencial: y = a * e^(bx)
        """
        try:
            # Transformación logarítmica: ln(y) = ln(a) + bx
            # Solo usar valores positivos de y
            mask = self.y > 0
            if not np.any(mask):
                raise ValueError("No hay valores positivos en y para aproximación exponencial")

            x_pos = self.x[mask]
            y_pos = self.y[mask]
            ln_y = np.log(y_pos)

            # Regresión lineal en el espacio transformado
            sum_x = np.sum(x_pos)
            sum_ln_y = np.sum(ln_y)
            sum_x_ln_y = np.sum(x_pos * ln_y)
            sum_x2 = np.sum(x_pos ** 2)
            n = len(x_pos)

            # Coeficientes
            b = (n * sum_x_ln_y - sum_x * sum_ln_y) / (n * sum_x2 - sum_x ** 2)
            ln_a = (sum_ln_y - b * sum_x) / n
            a = np.exp(ln_a)

            # Función exponencial
            def f_exp(x):
                return a * np.exp(b * x)

            # Coeficiente de correlación
            r2 = self.coeficiente_correlacion(self.y, f_exp(self.x))

            return {
                'coeficientes': {'a': a, 'b': b},
                'funcion': f_exp,
                'ecuacion': f'y = {a:.4f} * e^({b:.4f}x)',
                'r_cuadrado': r2
            }

        except Exception as e:
            # Método alternativo usando curve_fit
            def exp_func(x, a, b):
                return a * np.exp(b * x)

            try:
                popt, _ = curve_fit(exp_func, self.x, self.y, maxfev=5000)
                a, b = popt

                def f_exp(x):
                    return a * np.exp(b * x)

                r2 = self.coeficiente_correlacion(self.y, f_exp(self.x))

                return {
                    'coeficientes': {'a': a, 'b': b},
                    'funcion': f_exp,
                    'ecuacion': f'y = {a:.4f} * e^({b:.4f}x)',
                    'r_cuadrado': r2
                }
            except:
                return {
                    'error': 'No se pudo realizar la aproximación exponencial',
                    'coeficientes': None,
                    'funcion': None,
                    'ecuacion': 'Error en aproximación',
                    'r_cuadrado': 0
                }

    def coeficiente_correlacion(self, y_real, y_pred):
        """
        Calcula el coeficiente de determinación R²
        """
        ss_res = np.sum((y_real - y_pred) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def graficar_aproximaciones(self, mostrar_polinomial=True, grado_poly=2):
        """
        Grafica los datos originales y las aproximaciones
        """
        plt.figure(figsize=(15, 5))

        # Datos para graficar las funciones
        x_plot = np.linspace(min(self.x), max(self.x), 100)

        # Obtener aproximaciones
        lineal = self.aproximacion_lineal()
        exponencial = self.aproximacion_exponencial()

        # Subplot 1: Aproximación Lineal
        plt.subplot(1, 3, 1)
        plt.scatter(self.x, self.y, color='red', label='Datos originales', s=50)
        plt.plot(x_plot, lineal['funcion'](x_plot), 'b-', label=f'Lineal (R²={lineal["r_cuadrado"]:.4f})')
        plt.title('Aproximación Lineal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Aproximación Polinomial
        if mostrar_polinomial:
            plt.subplot(1, 3, 2)
            polinomial = self.aproximacion_polinomial(grado_poly)
            plt.scatter(self.x, self.y, color='red', label='Datos originales', s=50)
            plt.plot(x_plot, polinomial['funcion'](x_plot), 'g-',
                     label=f'Polinomial grado {grado_poly} (R²={polinomial["r_cuadrado"]:.4f})')
            plt.title(f'Aproximación Polinomial (grado {grado_poly})')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Subplot 3: Aproximación Exponencial
        plt.subplot(1, 3, 3)
        plt.scatter(self.x, self.y, color='red', label='Datos originales', s=50)
        if 'error' not in exponencial:
            plt.plot(x_plot, exponencial['funcion'](x_plot), 'm-',
                     label=f'Exponencial (R²={exponencial["r_cuadrado"]:.4f})')
        else:
            plt.text(0.5, 0.5, 'Error en aproximación\nexponencial',
                     transform=plt.gca().transAxes, ha='center', va='center')
        plt.title('Aproximación Exponencial')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def resumen_aproximaciones(self, grado_poly=2):
        """
        Muestra un resumen de todas las aproximaciones
        """
        print("=" * 60)
        print("RESUMEN DE APROXIMACIONES POR MÍNIMOS CUADRADOS")
        print("=" * 60)

        # Aproximación lineal
        lineal = self.aproximacion_lineal()
        print(f"\n1. APROXIMACIÓN LINEAL:")
        print(f"   Ecuación: {lineal['ecuacion']}")
        print(f"   R² = {lineal['r_cuadrado']:.6f}")

        # Aproximación polinomial
        polinomial = self.aproximacion_polinomial(grado_poly)
        print(f"\n2. APROXIMACIÓN POLINOMIAL (grado {grado_poly}):")
        print(f"   Ecuación: {polinomial['ecuacion']}")
        print(f"   R² = {polinomial['r_cuadrado']:.6f}")

        # Aproximación exponencial
        exponencial = self.aproximacion_exponencial()
        print(f"\n3. APROXIMACIÓN EXPONENCIAL:")
        if 'error' not in exponencial:
            print(f"   Ecuación: {exponencial['ecuacion']}")
            print(f"   R² = {exponencial['r_cuadrado']:.6f}")
        else:
            print(f"   {exponencial['error']}")

        print("\n" + "=" * 60)


# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 14.2, 16.8, 18.9, 21.1]

    # Crear objeto de mínimos cuadrados
    mc = MinimosCuadrados(x, y)

    # Mostrar resumen
    mc.resumen_aproximaciones(grado_poly=2)

    # Graficar aproximaciones
    mc.graficar_aproximaciones(mostrar_polinomial=True, grado_poly=2)

    # Ejemplo con datos exponenciales
    print("\n" + "=" * 60)
    print("EJEMPLO CON DATOS EXPONENCIALES")
    print("=" * 60)

    x_exp = np.array([0, 1, 2, 3, 4, 5])
    y_exp = np.array([1, 2.7, 7.4, 20.1, 54.6, 148.4])

    mc_exp = MinimosCuadrados(x_exp, y_exp)
    mc_exp.resumen_aproximaciones(grado_poly=2)
    mc_exp.graficar_aproximaciones(mostrar_polinomial=True, grado_poly=2)