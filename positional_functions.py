import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle5
import json


def get_distance(p1, p2):
    return np.linalg.norm(np.asarray(p1) - np.asarray(p2))

def plot_positions(list_positions, safe_threshold):
  fig = plt.figure(figsize=(30,15))
  ax = fig.gca()
  for a, b in combinations(list_positions,2):
      j = np.linspace(a[0], b[0], 100)
      k = np.linspace(a[1], b[1], 100)
      d = get_distance(a,b)
      s = "%.2f" % d
      m = [(a[i]+b[i])/2. for i in (0, 1)]
      clr = 'g'
      if d < safe_threshold:
        clr = 'r'
        ax.plot(j, k, clr)
        #ax.text(m[0], m[1], s)
  ax.scatter(*zip(*list_positions), c='b')
  ax.legend([Line2D([0], [0], marker='o', color='w', label='Circle', markerfacecolor='b', markersize=8)],["Count: "+str(len(list_positions))], loc='upper left')
  return fig

def pixels_to_cm(pixels, conv_scale):
    return pixels * conv_scale

centers = pickle5.load(open("yolocenters" , "rb"))

allframe = []
for fnum, info in centers.items():
    fr = []
    person = 1
    allframe.append(info)

framenum = 386
frpts = framenum * 2
print(allframe[framenum])

fig = plot_positions(allframe[framenum],400)

plt.show()