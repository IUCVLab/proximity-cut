import numpy as np
from numpy.linalg import norm as vector_norm
from sklearn.metrics.pairwise import cosine_similarity as cos
from nsw.nsw import Node, NSWGraph
from scipy.interpolate import Rbf

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gaussian_rbf(p, x, eps):
    r = vector_norm(p - x)
    return np.exp(-(eps * r) ** 2)


def edge_cos(a, b, x, eps):
    c = (a + b) / 2
    dx = x - c
    cs = cos(dx.reshape(1, -1), (b - c).reshape(1, -1))[0][0]
    val = cs / np.exp(vector_norm(dx))
    return val


def simple_dot(a, b, x, eps):
    c = (a + b) / 2
    dx = x - c
    return 2. * sigmoid(np.dot(dx, c)) - 1.


def amplified_cos(a, b, x, eps):
    c = (a + b) / 2
    dx = x - c
    de = a - b
    return 2. * sigmoid(np.dot(dx, de) / np.dot(de, de) * np.dot(dx, dx)) - 1.


def get_builtin_potential_function(G, cut, eps):
    points = []
    values = []
    for e, length in cut:
        a, b = G.nodes[e[0]].value, G.nodes[e[1]].value
        c = (a + b) / 2
        points.append(c)
        values.append(1)
    points = np.array(points)
    f = Rbf(points[:, 0], points[:, 1], np.array(values), 
            function="gaussian",
            epsilon=eps)
    return f


def get_rbf_potential_function(G, cut):
    def f(x, eps=10):
        result = 0.0
        for e, length in cut:
            a, b = G.nodes[e[0]].value, G.nodes[e[1]].value
            c = (a + b) / 2
            result += gaussian_rbf(c, x, eps)   
        return result    
    return f 


def get_cos_potential_function(G, cut):
    def f(x, eps=10):
        result = 0.0
        for e, length in cut:
            a, b = G.nodes[e[0]].value, G.nodes[e[1]].value
            if G.nodes[e[0]]._class > G.nodes[e[1]]._class:
                a, b = b, a
            # c = (a + b) / 2
            result += edge_cos(a, b, x, eps)           
        return result    
    return f 


def get_simple_dot_potential_function(G, cut):
    def f(x, eps=10):
        result = 0.0
        for e, length in cut:
            a, b = G.nodes[e[0]].value, G.nodes[e[1]].value
            if G.nodes[e[0]]._class > G.nodes[e[1]]._class:
                a, b = b, a
            result += simple_dot(a, b, x, eps)           
        return result    
    return f 


def get_amplified_cos_potential_function(G, cut):
    def f(x, eps=10):
        result = 0.0
        for e, length in cut:
            a, b = G.nodes[e[0]].value, G.nodes[e[1]].value
            if G.nodes[e[0]]._class > G.nodes[e[1]]._class:
                a, b = b, a
            result += amplified_cos(a, b, x, eps)           
        return result    
    return f 


def get_gaussian_rbf_grad(G, cut):
    def f(x, eps=10):
        result = np.zeros(x.shape)
        for e, length in cut:
            a, b = G.nodes[e[0]].value, G.nodes[e[1]].value
            c = (a + b) / 2
            k = gaussian_rbf(c, x, eps)
            result += k * (x - c) 
        return result * (- 2 * eps ** 2)    
    return f 
