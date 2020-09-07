import numpy as np
import random
from numpy.linalg import norm as vector_norm

def gen(N=400, border=0.8):
    values = []
    for i in range(N):
        p = np.array([1 - 2 * random.random(), 1 - 2 * random.random()])
        cls = 1 if vector_norm(p) > border else 0
        values.append((p, cls))
    return values

def gen2(N=400, border1=0.3, border2=0.9):
    values = []
    for i in range(N):
        p = np.array([1 - 2 * random.random(), 1 - 2 * random.random()])
        cls = 1 if border2 > vector_norm(p) > border1 else 0
        values.append((p, cls))
    return values

def gen_kd(N=400, k=3, border=0.3, noise=0.0):
    values = []
    for i in range(N):
        p = np.array([1 - 2 * random.random() for _ in range(k)])
        cls = 1 if vector_norm(p) > border else 0
        if random.random() < noise:
            cls = 1 - cls
        values.append((p, cls))
    return values


if __name__ == "__main__":
    print("Module data_gen launched as program.")