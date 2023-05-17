import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

filename = 'world0/testlog.txt'
filename1 = 'world0/envlog.txt'
x, y, vx, vy, v, t, obx, oby = [], [], [], [],[], [], [], []
index = []
j = 0
with open(filename, 'r') as f:
    for l in f:
        l = [e.strip() for e in l.split(',')]
        x.append(float(l[0]))
        y.append(float(l[1]))
        vx.append(float(l[2]))
        vy.append(float(l[3]))
        v.append((float(l[2])**2 + float(l[3])**2)**0.5)
        index.append(j)
        j=j+1
        t.append(0.5)

with open(filename1, 'r') as f:
    for l in f:
        l = [e.strip() for e in l.split(',')]
        obx.append(float(l[0]))
        oby.append(float(l[1]))

plt.figure(figsize=(8,8))
plt.plot(index, vx, color = 'red', alpha=0.8)
plt.plot(index, vy,color = 'blue')
plt.plot(index, v,color = 'green')
plt.grid(True)
plt.legend(('vx','vy','v'))

fig = plt.figure(figsize=(12, 12))
ax=fig.gca(projection='3d')
u=np.linspace(0,2*np.pi,40)
h=np.linspace(0,1,2)
for i in range(len(obx)):
    o=np.outer(obx[i] + 0.2 * np.sin(u),np.ones(len(h)))
    p=np.outer(oby[i] + 0.2 * np.cos(u),np.ones(len(h)))
    q=np.outer(np.ones(len(u)),h)
    ax.plot_surface(o,p,q,color='grey',linewidth=2,alpha=0.4)
    ax.scatter(obx[i],oby[i],0,'o',s=170,c="grey",alpha=0.9)
ax.plot(x, y, t,color='red',linewidth=3,alpha=1) 
ax.plot(x, y, 0,color='black',linestyle='dotted',linewidth=2,alpha=0.8) 
ax.scatter(0,0,0.5,'*',s=50,c='blue')
ax.scatter(10,10,0.5,'s',s=200,c='yellow')
ax.view_init(elev=55,azim=250)
plt.gca().set_box_aspect((1,1,0.2))
ax.set_xlabel('Position X (m)',fontsize='15')
ax.set_ylabel('Position Y (m)',fontsize='15')
ax.set_zlabel('Position Z (m)',fontsize='15')
ax.tick_params(labelsize=10)
#ax.set_title('flight trajectory')
plt.legend(('trajectory', 'projectory of trajectory'),loc='upper left',fontsize='20')
plt.show()
