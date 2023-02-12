import rospy
import torch
from geometry_msgs.msg import Vector3
from simenv import sim_env
from geometry_msgs.msg import Twist
from Agent import DQNAgent
import numpy as np

action_dict = {0: [-1.0, -1.0], 1: [-1.0, 0.0], 2: [-1.0, 1.0],
               3: [0.0, -1.0], 4: [0.0, 0.0], 5: [0.0, 1.0],
               6: [1.0, -1.0], 7: [1.0, 0.0], 8: [1.0, 1.0]}

max_episode = 40000
fc_n = 256
e_greedy = 0.1
buff_length = 2000
lr = 0.01
batch_size = 32
gamma = 0.99
update_time = 20
n_states = 4
n_acts = 9
max_step_per_eps = 300

def state_processing(data):
    state = np.zeros((n_states), dtype=np.float)
    state[0] = data.pose.position.x
    state[1] = data.pose.position.y
    state[2] = data.twist.linear.x
    state[3] = data.twist.linear.y
    return state

def act_processing(data):
    ctrl = action_dict.get(data)
    velocity = Twist()
    velocity.linear.x = ctrl[0]
    velocity.linear.y = ctrl[1]
    return velocity

rospy.init_node('simenv_node', anonymous=True)
goal = Vector3()
goal.x = 3
goal.y = 3
goal.z = 1
env = sim_env(goal, max_step_per_eps)
agent = DQNAgent(n_states, n_acts, fc_n, e_greedy, lr, gamma, buff_length, batch_size, update_time)
buf_cnt = 0
total_reward = 0.0
start = False
rospy.loginfo('get exp data...')
for episode in range(max_episode):
    drone_state = env.reset()
    state = state_processing(drone_state)
    total_reward = 0.0
    while True:
        action = agent.choose_action(state)
        u = act_processing(action)
        drone_state, reward, done = env.step(u)
        state_ = state_processing(drone_state)
        buf_cnt += 1
        agent.store_data(state, action, reward, state_, done, buf_cnt)
        state = state_
        if start == True:
            agent.learn()
        if buf_cnt >= buff_length and start == False:
            start = True
            print('start train')
        total_reward += reward
        if(done):
            break
    if start == True:
        print('episode: ', episode, 'total reward: ', total_reward)
    if start == True and (episode + 1) % 400 == 1:
        torch.save(agent.q_net,  'model.ckpt')


