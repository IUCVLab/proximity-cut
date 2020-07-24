import numpy as np
from numpy.linalg import norm as vector_norm
from sklearn.metrics.pairwise import cosine_similarity as cos
from nsw.nsw import Node, NSWGraph
from scipy.interpolate import Rbf
from scipy.special import erf
from sortedcontainers import SortedList

def sigmoid(x):
    '''Simple REAL sigmoid function to bring function to [0 .. 1] interval '''
    return 1 / (1 + np.exp(-x))


def signed_sigmoid(x):
    '''Simple REAL sigmoid function to bring function to [-1 .. 1] interval '''
    return 2 / (1 + np.exp(-x)) - 1.


def gaussian_rbf(p, x, eps):
    '''REAL Gaussian radial basis function. Equals 1 in `p` and 0 far from `p` '''
    r = vector_norm(p - x)
    return np.exp(-(eps * r) ** 2)


def gaussian_vector_rbf(a, b, x, eps):
    '''VECTOR Gaussian radial basis function. Equals `1/exp(b - a)` in point `(a+b)/2` and 0 far from it.'''
    center = (a + b) / 2
    edge = b - a
    dx = x - center
    edge_length = vector_norm(edge)
    norm = np.exp(edge_length)
    r = vector_norm(dx)
    return (edge / edge_length) * np.exp(-(eps * r) ** 2) / edge_length / norm


def edge_cos(a, b, x, eps):
    '''REAL cosine similarity fuction with fade out. For `c=(a+b)/2` equals `cos_sim(x-c, b-c)/exp(norm(x-c))`. Positive where `ab` and `x` lay in the same semispace.'''
    c = (a + b) / 2
    dx = x - c
    cs = cos(dx.reshape(1, -1), (b - c).reshape(1, -1))[0][0]
    val = cs / np.exp(vector_norm(dx))
    return val


def simple_dot(a, b, x, eps):
    '''REAL sigmoid or dot product. For `c=(a+b)/2` equals `sgm(x-c, b-c)` . Lay in `[-1 .. 1]`'''
    c = (a + b) / 2
    dx = x - c
    return 2. * sigmoid(np.dot(dx, b-c)) - 1.


def amplified_cos(a, b, x, eps):
    '''Amplified REAL function, similar to `edge_cos`, but faster to compute. Lay in `[-1, 1]`.'''
    c = (a + b) / 2
    dx = x - c
    de = a - b
    return 2. * sigmoid(np.dot(dx, de) / np.dot(de, de) * np.dot(dx, dx)) - 1.


def get_builtin_potential_function(G, cut, eps):
    '''Returns `scipy.Rbf` for graph cut centers.'''
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
    '''Returns REAL Gaussian-based RBF approximation function of class border. Bigger where the border is'''
    def f(x, eps=10):
        result = 0.0
        for e, length in cut:
            a, b = G.nodes[e[0]].value, G.nodes[e[1]].value
            c = (a + b) / 2
            result += gaussian_rbf(c, x, eps)   
        return result    
    return f


def get_vector_rbf_potential_function(G, cut):
    '''Returns VECTOR Gaussian-based approximation function. Is an approximation of `grad(classifier)`.'''
    def f(x, eps=10):
        result = 0.0
        for e, length in cut:
            a, b = G.nodes[e[0]].value, G.nodes[e[1]].value
            if G.nodes[e[0]]._class > G.nodes[e[1]]._class:
                a, b = b, a
            result += gaussian_vector_rbf(a, b, x, eps)   
        return result    
    return f 


def get_cos_potential_function(G, cut):
    '''Returns REAL cos-based function. Is a classifier approximation.'''
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
    '''Returns REAL `simple_dot`-based function. Is a classifier approximation.'''
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
    '''Returns REAL `amplified_cos`-based function. Is a classifier approximation.'''
    def f(x, eps=10):
        result = 0.0
        for e, length in cut:
            a, b = G.nodes[e[0]].value, G.nodes[e[1]].value
            if G.nodes[e[0]]._class > G.nodes[e[1]]._class:
                a, b = b, a
            result += amplified_cos(a, b, x, eps)           
        return result    
    return f 


def get_grad_field_function(G, cut):    
    def f(x, eps=0.05):
        # step 1. Get closest edge points
        # todo: build an index
        cl = SortedList()
        for e, length in cut:
            a, b = G.nodes[e[0]].value, G.nodes[e[1]].value
            center = (a + b) / 2
            if G.nodes[e[0]]._class > G.nodes[e[1]]._class: a, b = b, a
            
            vect = (b - a) / vector_norm(b - a)
            d = np.dot(center - x, center - x)
            cl.add((d, vect))

        vectors = np.zeros(cl[0][1].shape)
        sum = 0.
        for d, vect in cl:
            if d > eps:
                break
            c = 2 / (1 + d / eps) - 1
            vectors += c * vect
            sum += 1
        if sum > 0:
            vectors /= sum
            
        return vectors
    return f
    

def get_grad_based_classifier_function(G, cut):
    '''Returns REAL -1..1 value of belonging to a class. Based on [Gradient theorem](https://en.wikipedia.org/wiki/Gradient_theorem)'''
    
    grad = get_grad_field_function(G, cut)
    
    def f(x, eps=.005, M=10):
        # step 2. Get integral along (d0 .. x) vector
        result = 0.
        for dd0, gtc in [([.5, .5], 1), ([.62, .55], 1), ([0., 0], -1), ([.0, .1], -1)]:
            dd0 = np.array(dd0)
            r = x - dd0
            for i in range(M):
                p = dd0 + (r * i / M)
                result += np.dot(grad(p, eps), r) * gtc
        print(".", end="")
        return result
    return f