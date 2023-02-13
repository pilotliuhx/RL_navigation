#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Transform
from trajectory_msgs.msg import MultiDOFJointTrajectory
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
class sim_env():
    def __init__(self, goal_pos, max_steps):
        self.drone_states = ModelState()
        self.goal = goal_pos
        self.vel_publisher = rospy.Publisher('/iris/command/trajectory', MultiDOFJointTrajectory, queue_size=10)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_states_callback)
        self.ctrl_rate = rospy.Rate(50)
        self.max_steps = max_steps
        self.step_cnt = 0
        self.distance_ = 0.0
        self.action_ = Twist()
        self.set_goal()

    def gazebo_states_callback(self, data):
        self.drone_states.pose = data.pose[1]
        self.drone_states.twist = data.twist[1]
        #rospy.loginfo('received a state data!')
        
    def get_reward(self, vel):
        done = False
        reward = 0.0
        dis_x = self.drone_states.pose.position.x - self.goal.x
        dis_y = self.drone_states.pose.position.y - self.goal.y
        distance = math.sqrt(dis_x ** 2 + dis_y ** 2)
        
        reward =  (self.distance_ - distance)
        reward -= 0.05 * (abs(self.action_.linear.x - vel.linear.x) + abs(self.action_.linear.y - vel.linear.y))

        self.distance_ = distance
        self.action_ = vel

        if(distance < 0.1):
            reward += 5
            done = True
        if self.step_cnt >= self.max_steps:
            done = True
        return reward, done

    def step(self, vel):
        self.step_cnt += 1
        target_pos = Transform()
        target_pos.translation.z = 1.0
        target_acc = Twist()
        vel_cmd = MultiDOFJointTrajectory()
        point = MultiDOFJointTrajectoryPoint()
        point.transforms.append(target_pos)
        point.velocities.append(vel)
        point.accelerations.append(target_acc)
        vel_cmd.header.stamp = rospy.Time.now()
        vel_cmd.points.append(point)
        self.vel_publisher.publish(vel_cmd)
        #rospy.loginfo('published a message!')
        self.ctrl_rate.sleep()
        reward, done = self.get_reward(vel)
        # rospy.loginfo('model pos : %f, %f, %f' ,self.drone_states.pose.position.x, \
        # self.drone_states.pose.position.y, self.drone_states.pose.position.z)
        return self.drone_states, reward, done
    
    def set_goal(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        m_set_srv = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        state = ModelState()
        state.model_name = 'destination'
        state.pose.position.x = self.goal.x
        state.pose.position.y = self.goal.y
        state.pose.position.z = 1.0
        m_set_srv(state)

    def reset(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        h_reset_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state = ModelState()
        state.model_name = 'iris'
        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 1.0
        self.step_cnt = 0
        h_reset_srv(state)
        self.ctrl_rate.sleep()
        dis_x = self.drone_states.pose.position.x - self.goal.x
        dis_y = self.drone_states.pose.position.y - self.goal.y
        self.distance_ = math.sqrt(dis_x ** 2 + dis_y ** 2)
        # rospy.loginfo('model reset to : %f, %f, %f' ,self.drone_states.pose.position.x, \
        # self.drone_states.pose.position.y, self.drone_states.pose.position.z)
        return self.drone_states

if __name__ == '__main__':
    rospy.init_node('simenv_node', anonymous=True)
    goal = Vector3()
    goal.x = 0
    goal.y = 10
    goal.z = 1
    env = sim_env(goal)
    rate = rospy.Rate(10)
    velicity = Twist()
    velicity.linear.x = 0.1
    velicity.linear.y = 0.1
    velicity.linear.z = 0.0
    timecnt = 0
    while not rospy.is_shutdown():
        timecnt += 1
        try:
            if timecnt > 500:
                timecnt = 0
                env.reset()
            env.step(velicity)
            #rate.sleep()
        except rospy.ROSInterruptException:
            pass
