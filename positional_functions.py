import numpy as np
from itertools import combinations


def get_distance(p1, p2):
    return np.linalg.norm(np.asarray(p1) - np.asarray(p2))


def get_all_distances(list_points):
    return {(p[0], p[1]): get_distance(p[0], p[1]) for p in combinations(list_points, 2)}


def pixels_to_cm(pixels, conv_scale):
    return pixels * conv_scale


listpts = [(1, 2), (2, 1), (2, 3)]
print(get_all_distances(listpts))
