import matplotlib.pyplot as plt
import torch
import math
import numpy as np
import functools
def smoothstep(x, edge0=0, edge1=1):
    x = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return 3 * x ** 2 - 2 * x ** 3
def quintic(x):
    return 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3
def cubic(x):
    return 3 * x ** 2 - 2 * x ** 3
def lerp(a, b, x):
    return (1-x)*a+x*b
def hash(x): #value in [-1,1]
    k = 0x456789ab
    x_ = torch.empty(x.size())
    for i, n in enumerate(x.int()):
        n ^= (n << 24)
        n ^= (n >> 1)
        n ^= (n << 1)
        n *= k
        x_[i] = 2 * (float(n) % 0xffffffff / 0xffffffff) - 1
    return x_
def vnoise(x, freq=10, phase=0):
    x = x + phase
    x = x * freq
    i = torch.floor(x)
    f = torch.frac(x)
    a = hash(i)
    b = hash(i + 1)
    return torch.lerp(a, b, quintic(f))
def vnoise_(x, freq=10, phase=0):
    x = x * freq + phase
    i = torch.floor(x)
    f = torch.frac(x)
    a = hash(i)
    b = hash(i+1)
    return torch.lerp(a, b, smoothstep(f))
def gnoise(x, freq=10, phase=0):
    x = x + phase
    x = x * freq
    i = torch.floor(x)
    f = torch.frac(x)
    a = hash(i)
    b = hash(i + 1)
    return torch.lerp(a*f, b*(1-f), quintic(f))
def fbm(x, H=0.5, freq=4, phase=0, itr=4, func=vnoise):
    v = 0
    t = 1
    G = 2 ** -H
    for i in range(itr):
        v = v + t * func(x, freq=freq, phase=phase)
        t = t * G
        freq = freq * 1.99
    return v
def warp(x, func, itr=3, wt=.5):
    for i in range(itr):
        x = x + wt*func(x)
    return func(x)
def sin(x, H=1., freq=4, phase=0, itr=4):
    v = 0
    t = 1
    G = 2 ** -H
    for i in range(itr):
        v = v + t * torch.sin(math.pi*(x+phase)* freq)
        t = t * G
        freq = freq * 1.99 
    return v


def sin_(x, freq=4, phase=0):
    v = torch.sin(math.pi*(x+phase)* freq)
    return v
# sequence function


def shade(n, step=1):
    a = torch.arange(1, 1+n//2, step)
    a_ = n-a
    a_ = a_.flip(0)
    b = torch.cat([a, a_], 0).reshape(1, -1)
    c = torch.cat([sym(b), sym(b.flip(1))], 0).reshape(1, -1)
    return c


def sym(s):
    c = torch.cat([s, s.flip(1)], 0)
    c = c.transpose(0, 1).flatten()
    return c


def reverse(s):
    zero = torch.tensor([[0]])
    return torch.cat([zero, s], dim=1)


def dotSeq(t, lng=9, itr=1):
    seq = list(t.squeeze())
    for i in range(itr):
        nlist = []
        for j in list(seq):
            if j < lng:
                nlist.append(j)
            else:
                nlist += [math.ceil(j / 2),
                          1,
                          j - math.ceil(j / 2) - 1]
        seq = nlist
    return torch.tensor(nlist, dtype=torch.long).reshape(1, -1)
