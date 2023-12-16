import rospy
import math
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan

x = []
y = []


def lidar_states_callback(lidar_data):
    global x
    global y

    r = np.array(lidar_data.ranges)

    num_points = int(math. ceil((lidar_data.angle_max - lidar_data.angle_min) / lidar_data.angle_increment))
    
    x = np.zeros(num_points)
    y = np.zeros(num_points)

    for i in range(num_points):
        theta = lidar_data.angle_min + lidar_data.angle_increment * i
        x[i] = -r[i] * np.cos(theta)
        y[i] = -r[i] * np.sin(theta)

    x[-1] = -r[-1] * np.cos(lidar_data.angle_max)
    y[-1] = -r[-1] * np.sin(lidar_data.angle_max)

    print(x)

if __name__ == '__main__':
    rospy.init_node('lidar_listener')
    rospy.Subscriber('/iris/scan', LaserScan, lidar_states_callback)

    fig, ax = plt.subplots()

    while not rospy.is_shutdown():
        ax.cla
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.grid(True)
        ax.scatter(y, x, s = 0.1)
        plt.draw()
        plt.pause(0.1)

    rospy.spin()
