from math import *
import sympy as sp
import matplotlib.pyplot as plot
from sympy.plotting import plot

def PolinomioTaylor(a,n):
    x = sp.symbols('x') #Definicion de la variable simbolica x
    f = sp.cos(x) #Funcion que vamos a trabajar
    F=f #Guardamos la funcion original (copia)
    taylor = f.subs(x,a)
    for k in range(1,n+1): 
        derivadak = sp.diff(f,x) #Derivada k-esima
        taylor = taylor + derivadak.subs(x,a)*((x-a)**k)/factorial(k) #Se escribe la sumatoria del polinomio de Taylor
        f = derivadak

    print(sp.expand(taylor))
    g = plot(F,taylor, (x,a-3,a+3),title='Polinomio de Taylor de la función f(x)=exp(x)',show=False)
    g[0].line_color = 'blue' #Funcion F(x)
    g[1].line_color = 'red' #Funcion del polimomio de taylor
    g.show()
    plot.xlabel('x')
    plot.ylabel('y')
    plot.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plot.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plot.legend()
    plot.grid(True)
    plot.show()
    
a = float(input("Digite alrededor de que punto desea el polinomio de Taylor: "))
n = int(input("Digite el orden del polinomio de Taylor: "))
PolinomioTaylor(a,n)

