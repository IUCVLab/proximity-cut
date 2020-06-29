import numpy as np
import random
from numpy.linalg import norm as vector_norm

def gen(N=400, border=0.8):
    values = []
    for i in range(N):
        p = np.array([random.random(), random.random()])
        cls = 1 if vector_norm(p) > border else 0
        values.append((p, cls))
    return values

def gen2(N=400, border1=0.3, border2=0.9):
    values = []
    for i in range(N):
        p = np.array([random.random(), random.random()])
        cls = 1 if border2 > vector_norm(p) > border1 else 0
        values.append((p, cls))
    return values


if __name__ == "__main__":
    print("Module data_gen launched as program.")