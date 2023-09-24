from sympy import *
x = symbols('x')
print(latex(Integral(cos(x)**2, (x, 0, pi))))
