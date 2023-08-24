import os
from itertools import zip_longest
import numpy as np

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def transpose(l):
    toret=[list(i) for i in zip_longest(*l)]
    return toret




def scale(inarray, interval=[0,100]):
    m= min(inarray)
    M = max(inarray)
    new_min= interval[0]
    new_max= interval[1]
    inarray = np.array(inarray)
    return ((inarray - m) * (new_max - new_min)) / ((M - m) + new_min)

