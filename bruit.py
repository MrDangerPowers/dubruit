from random import uniform as unif
from math import *
import numpy as np
from PIL import Image


rotmat = np.array([[ 4/5, 3/5, 0],
                   [-3/5, 4/5, 0],
                   [   0,   0, 1]])


def sstep(x):
    return x * x * (3 - 2 * x)

def d2step(pt, a, b, c, d):
    x, y = pt
    ssx = sstep(x)
    ssy = sstep(y)
    s = a
    s += (b - a) * ssx
    s += (d - a) * ssy
    s += (a - b + c - d) * ssx * ssy
    return s


def randarray(dim):
    tab = np.ndarray(dim, float)
    for i in range(dim[0]):
        for j in range(dim[1]):
            tab[i, j] = unif(0, 1)

    return tab

class niveau:

    def __init__(self, depmat, coef, dim):
        self.depmat = depmat
        self.coef = coef
        self.dim = dim

    def __getitem__(self, pt):
        i, j = pt
        return np.matmul(np.array([i, j, 1]), self.depmat)[:2]

class noise_grid:

    def __init__(self, profondeur):
        t = round(1.5 * 2 ** (profondeur + 1)) 
        self.tab = randarray((t, t))
        self.niv = []
        self.sc = 0
        dim = 1
        rota = np.array([[     1,      0, 0],
                         [     0,      1, 0],
                         [t // 2, t // 2, 1]])
        for _ in range(profondeur):
            coef = 1 / dim
            self.sc += coef
            pmat = np.array([[        1,         0, 0],
                             [        0,         1, 0],
                             [-dim // 2, -dim // 2, 1]])
            self.niv.append(niveau(np.matmul(pmat, rota), coef, dim))
            dim *= 2
            rota = np.matmul(rotmat, rota)

    def __getitem__(self, pt):
        i, j = pt
        s = 0
        for niv in self.niv:
            x = i * niv.dim
            y = j * niv.dim
            x, y = niv[x, y]
            u = int(x)
            v = int(y)
            pts = []
            for k in range(2):
                for l in range(2):
                    pts.append(self.tab[u + l, v + k])
            a, b, d, c = pts
            s += niv.coef * d2step((x - u, y - v), a, b, c, d)

        return s / self.sc

def main():
    dim = (2000, 2000)
    im = Image.new('RGB', dim, (0, 0, 0))
    pix = im.load()

    a = noise_grid(8)
    a.niv[0].coef = 0
    a.sc -= 1
    for i in range(dim[0]):
        if i % 10 == 0:
            print(i)
        for j in range(dim[1]):
            x = i / dim[0]
            y = 1 - j / dim[1]
            t = a[x, y]
            h = int((t % 0.1) * 2550)
            k = int((t // 0.1) * 25.5)
            l = int((t % 0.05) * 255 * 20)
            pix[i, j] = (k, 0, abs(k - l))
    im.show()
    im.save('test.png')
    im.close()

main()
