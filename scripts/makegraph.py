import os
import numpy as np
import matplotlib.pyplot as plt

filename = "world2/log2.txt"
filename1 = "world2-1/log2-1.txt"
x, y = [], []
x1,y1 = [],[]
# extract out x's and y's
with open(filename, 'r') as f:
    for l in f:
        l = [e.strip() for e in l.split(':')]
        if 'Average Episodic Return' in l:
            y.append(float(l[1]))
        if 'Timesteps So Far' in l:
            x.append(int(l[1]))
with open(filename1, 'r') as f:
    for l in f:
        l = [e.strip() for e in l.split(':')]
        if 'Average Episodic Return' in l:
            y1.append(float(l[1]))
        if 'Timesteps So Far' in l:
            x1.append(int(l[1]))
plt.plot(x, y, 'b', alpha=0.8)
plt.plot(x1, y1, 'r', alpha=0.8)
plt.legend(['Without speed limitation', 'Limit speed to 5 m/s'])
plt.xlabel('Average Total Timesteps So Far')
plt.ylabel('Average Episodic Return')
# Show graph so user can screenshot
plt.show()