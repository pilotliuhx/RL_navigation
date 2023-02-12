import torch
import gym
from model import Net
from Agent import DQNAgent
import numpy as np
max_episode = 4000
fc_n = 128
e_greedy = 0.1
buff_length = 2000
lr = 0.01
batch_size = 200
gamma = 0.9
update_time = 20
game_name = 'MountainCar-v0'  #  MountainCar-v0  CartPole
env = gym.make(game_name)
env = env.unwrapped

n_states = env.observation_space.shape[0]
n_acts = env.action_space.n
agent = DQNAgent(n_states, n_acts, fc_n, e_greedy, lr, gamma, buff_length, batch_size, update_time)
buf_cnt = 0
total_reward = 0.0
start = False
print('get exp data')
for episode in range(max_episode):
    state = env.reset()
    total_reward = 0.0
    while True:
        action = agent.choose_action(state)
        state_, reward, done, _ = env.step(action)

        # x, x_dot, theta, theta_dot = state_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2
        
        r = state_[0] + 0.5
        if state_[0] > -0.5:
            r = state_[0] + 0.5
            if state_[0] > 0.5:
                r = 5
        else:
            r = 0

        reward = r
        buf_cnt += 1
        agent.store_data(state, action, reward, state_, done, buf_cnt)
        env.render()
        state = state_
        if start == True:
            agent.learn()
        if buf_cnt >= buff_length and start == False:
            start = True
            print('start train')
        if(done):
            break
        total_reward += reward
    if start == True:
        print('episode: ', episode, 'total reward: ', total_reward)
    if start == True and (episode + 1) % 400 == 1:
        torch.save(agent.q_net, game_name + 'model.ckpt')
        #agent.q_net.load_state_dict(torch.load('params.ckpt'))