import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 8)

def getCurvaLinedar(x, w, b=0, escalaRuido=0):
    return w*x + b + escalaRuido*np.random.randn(x.shape[0])

x = np.arange(-10, 30.1, 0.5)
Y = getCurvaLinedar(x, 1.8, 32, escalaRuido=2.5)


w = np.random.randn(1)
b = 0

def forward(entrada, w, b):
    return w*entrada + b

def mse(Y, y):
    return (Y-y)**2

# v = np.array([3, 4, 5])
# u = np.array([3.2, 4.5, 4.8])

def backpropagation(entradas, saidas, valorReal, w, b, lr):
    dw = lr*(-2*entradas*(valorReal-saidas)).mean()
    db = lr*(-2*(valorReal-saidas)).mean()

    w -= dw
    b-= db

    return w, b

def ajusteModelo(entradas, valorReal, w, b, epocas = 200, lr = 0.001):
    for epoca in range(epocas):
        saidas = forward(entradas, w, b)
        perda = np.mean(mse(valorReal, saidas))
        w, b = backpropagation(entradas, saidas, valorReal, w, b, lr)

        if (epoca + 1) % 50 == 0:
            print(f'Epoca: [{(epoca + 1)}/{epocas}] Perca: [{perda:.4f}]')

    return w, b

x = np.arange(-10,10,2)
Y = getCurvaLinedar(x, w=1.8, b=32, escalaRuido=2)

w = np.random.randn(1)
b = np.zeros(1)

w, b = ajusteModelo(x, Y, w, b, epocas=2000, lr=0.005)
print(f'w: {w[0]:.3f}, b: {b[0]:.3f}')


plt.scatter(x, Y)
plt.plot(x, getCurvaLinedar(x, w, b), 'r', lw=3)
plt.show()