from sklearn import datasets
import random

def get_mnist(random_seed=13):
    '''loads data and creates train/test splitting. Returns a tuple (train, test) where each item is a list of tuples (vector, class)'''
    random.seed(random_seed)
    digits = datasets.load_digits()
    X, Y = [x.flatten() for x in digits.images], digits.target
    XY = list(zip(X, Y))
    random.shuffle(XY)
    train_mnist, test_mnist = XY[:9 * len(XY) // 10], XY[9 * len(XY) // 10:]
    
    return train_mnist, test_mnist


