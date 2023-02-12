from model import Net
import torch
import numpy as np
import torch.nn as nn
class DQNAgent():
    def __init__(self, n_states, n_actions, n_fc, e_greedy, lr, gamma, buffer_length, batch_size, update_time) -> None:
        self.e_greedy = e_greedy
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_time = update_time
        self.lr = lr
        self.n_fc = n_fc
        self.n_states = n_states
        self.n_actions = n_actions
        self.buff_length = buffer_length
        self.q_net = Net(self.n_states, self.n_actions, self.n_fc)
        self.eval_net = Net(self.n_states, self.n_actions, self.n_fc)
        self.eval_net.load_state_dict(self.q_net.state_dict())
        self.buf_state = np.zeros([self.buff_length, self.n_states])
        self.buf_state_ = np.zeros([self.buff_length, self.n_states])
        self.buf_action = np.zeros(self.buff_length)
        self.buf_reward = np.zeros(self.buff_length)
        self.buf_done = np.zeros(self.buff_length)
        self.lear_cnt = 0
        self.lossf = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = self.lr)
        self.update_cnt = 0

    def load_net(self, model_name):
        self.q_net = torch.load( model_name)
        self.eval_net = torch.load( model_name)

    def choose_action(self, state):
        if np.random.random() < self.e_greedy:
            return np.random.choice(range(self.n_actions))
        else:
            return torch.argmax(self.q_net(torch.Tensor(state))).item()
    
    def store_data(self, state, action, reward, state_, done, buf_cnt):
        self.buf_state[buf_cnt % self.buff_length] = state
        self.buf_action[buf_cnt % self.buff_length] = action
        self.buf_reward[buf_cnt % self.buff_length] = reward
        self.buf_done[buf_cnt % self.buff_length] = done
        self.buf_state_[buf_cnt % self.buff_length] = state_

    def learn(self):
        batch_index = np.random.choice(self.buff_length, self.batch_size)
        batch_state = torch.Tensor(self.buf_state[batch_index, :])
        batch_state_ = torch.Tensor(self.buf_state_[batch_index, :])
        batch_reward = torch.Tensor(self.buf_reward[batch_index].reshape(-1,1))
        batch_action = torch.Tensor(self.buf_action[batch_index]).long().reshape(-1, 1)
        self.update_cnt += 1
        if self.update_cnt >= self.update_time:
            self.update_cnt = 0
            self.eval_net.load_state_dict(self.q_net.state_dict())
        current_Q = self.q_net(batch_state).gather(1, batch_action)
        q_eval = self.eval_net(batch_state_).detach().max(1)[0].reshape(self.batch_size, 1)
        target_Q = batch_reward + self.gamma * q_eval
        loss = self.lossf(current_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return
if __name__ == '__main__':
    agent = DQNAgent(4, 2, 128, 0.9)
    state = np.array([1, 1, 1, 1])
    action = agent.choose_action(state)
    print(action)
