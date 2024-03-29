"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without 
	relying on ppo.py.
"""
import numpy as np
from geometry_msgs.msg import Twist

def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def state_processing(data):
		state = np.zeros((4), dtype=np.float)
		state[0] = data.pose.position.x
		state[1] = data.pose.position.y
		state[2] = data.twist.linear.x
		state[3] = data.twist.linear.y
		return state
def act_processing(data):
	velocity = Twist()
	velocity.linear.x = data[0] * 1
	velocity.linear.y = data[1] * 1
	return velocity

def rollout(policy, env, render):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.

		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	test_log_flag = True
	# Rollout until user kills process
	while True:
		obs = state_processing(env.reset())
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		while not done:
			t += 1

			# Query deterministic action from policy and run it
			action = policy(obs).detach().numpy()
			action_in = act_processing(action)
			obs, rew, done = env.step(action_in)
			obs = state_processing(obs)
			# Sum all episodic rewards as we go along
			ep_ret += rew
			if test_log_flag == True:
				log_data = [obs[0],obs[1],obs[2],obs[3],action[0],action[1],rew]
				with open("/home/lhx/catkin_ws/src/simenv/scripts/testlog.txt","a") as logf:
					logf.write(str(log_data).replace('[',' ').replace(']',' ')+'\n')
		if test_log_flag == True:
			log_data = env.get_obstacle_pos()
			for pos in log_data:
				with open("/home/lhx/catkin_ws/src/simenv/scripts/envlog.txt","a") as logf:
					logf.write(str(pos.position.x) + ',' + str(pos.position.y) +'\n')
		test_log_flag = False
		# Track episodic length
		ep_len = t

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret

def eval_policy(policy, env, render=False):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)