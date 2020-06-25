import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


def show_state(points, highlight=None, leader=None, target=None, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.scatter(points[:, 0], points[:, 1], marker="*", color="orange")
    if highlight is not None:
        plt.scatter(highlight[:, 0], highlight[:, 1], marker="*", color="blue")
    if leader is not None:     
        plt.scatter(leader[:, 0], leader[:, 1], marker="o", color="black", s=100)
    if target is not None:     
        plt.scatter(target[:, 0], target[:, 1], marker="o", color="red", s=100)
    plt.show()