import os
import numpy as np
import matplotlib.pyplot as plt

filename = "world0/log0.txt"
x, y = [], []

# extract out x's and y's
with open(filename, 'r') as f:
    for l in f:
        l = [e.strip() for e in l.split(':')]
        if 'Average Episodic Return' in l:
            y.append(float(l[1]))
        if 'Timesteps So Far' in l:
            x.append(int(l[1]))
plt.plot(x, y, 'b', alpha=0.8)
plt.xlabel('Average Total Timesteps So Far')
plt.ylabel('Average Episodic Return')
# Show graph so user can screenshot
plt.show()