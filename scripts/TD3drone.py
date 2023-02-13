import rospy
import torch
import TD3
import utils
from geometry_msgs.msg import Vector3
from simenv import sim_env
from geometry_msgs.msg import Twist
import numpy as np

state_dim = 4
action_dim = 2
max_step_per_eps = 300
max_action = 2
test_period = 100
args = {
    'start_timesteps':1e3,
    'eval_freq': 5e3,
    'expl_noise': 0.1,
    'batch_size': 128,
    'discount': 0.99,
    'tau': 0.005,
    'policy_noise': 0.2,
    'noise_clip': 0.5,
    'policy_freq': 2,
    'lr': 1e-3
}

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": 1.0,
    "discount": args['discount'],
    "tau": args['tau'],
    "lr": args['lr']
}
def state_processing(data):
    state = np.zeros((state_dim), dtype=np.float)
    state[0] = data.pose.position.x
    state[1] = data.pose.position.y
    state[2] = data.twist.linear.x
    state[3] = data.twist.linear.y
    return state

def act_processing(data):
    velocity = Twist()
    velocity.linear.x = data[0] * max_action
    velocity.linear.y = data[1] * max_action
    return velocity

def params_callback(data):
    global args
    global test_period
    if data.x > 1e-5:
        args['expl_noise'] = data.x
    if data.y > 1e-5:
        test_period = data.y

rospy.init_node('simenv_node', anonymous=True)
rospy.Subscriber('/TD3/params', Vector3, params_callback)
goal = Vector3()
goal.x = 5
goal.y = 5
goal.z = 1
env = sim_env(goal, max_step_per_eps)
kwargs["policy_noise"] = args['policy_noise'] * max_action
kwargs["noise_clip"] = args['noise_clip'] * max_action
kwargs["policy_freq"] = args['policy_freq']

policy = TD3.TD3(**kwargs)
rospy.loginfo('policy is ready!')
#policy.load('TD3models/Atti_Models/history/ctrlmodel_8_199999')
replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
stepcounter = 0
rewardlog = []
start_train = False
start_data_collect = False
for episode in range(500000):
    drone_state = env.reset()
    state = state_processing(drone_state)
    #rospy.loginfo('state: %f %f %f %f', state[0], state[1], state[2], state[3])
    done = False
    episode_reward = 0
    episode_timesteps = 0
    if episode % test_period == 0:
        print('test episode: ')
    for step in range(max_step_per_eps):
        stepcounter += 1
        # generate action
        if stepcounter < args['start_timesteps']:
            if start_data_collect == False:
                print('start data collecting!')
                start_data_collect = True
            action = np.random.uniform(-1, 1, action_dim)
            action_in = act_processing(action)
        else:
            if start_train == False:
                print('start training!')
                start_train = True
            noise = np.random.normal(0, 1.0 * args['expl_noise'], size=action_dim).clip(-1.0, 1.0)
            action = policy.select_action(state)
            if episode % test_period != 0:
                action = action + noise
            action_in = act_processing(action)
        # perform action
        next_drone_state, reward, done = env.step(action_in)
        next_state = state_processing(next_drone_state)
        
        # store replay buffer
        if episode % test_period != 0:
            replay_buffer.add(state, action, next_state, reward, done)
        
        state = next_state
        episode_reward += reward
        if done: 
            break
        
        if stepcounter > args['start_timesteps']:
            if episode % test_period != 0:
                policy.train(replay_buffer, args['batch_size'])
            
    # End of Episode
    rewardlog.append(episode_reward)
    
    print('episode:', episode, 'reward:', episode_reward, 'step:', step)
    with open("/home/lhx/catkin_ws/src/simenv/scripts/rewardlog.txt","a") as logf:
        logf.write(str(episode) + " " + str(episode_reward) + "\n")
    if episode % 2000 == 1999:
        policy.save('ctrlmodel_'+str(episode))
        print ('Model', episode, 'saved!')
