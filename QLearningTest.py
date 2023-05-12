import numpy as np
import gym
import random

def update_q_table(prev_state, action, reward, nextstate, alpha, gamma):
    return (1 - alpha) * q_table[prev_state][action] + alpha * (reward + gamma * np.max(q_table[nextstate]))

def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # take random action
    else:
        return np.argmax(q_table[state])  # take action according to the q table

n_bins = 10
position_bins = np.linspace(-2.4, 2.4, num=n_bins)
velocity_bins = np.linspace(-3.0, 3.0, num=n_bins)
angle_bins = np.linspace(-0.5, 0.5, num=n_bins)
angular_velocity_bins = np.linspace(-2.0, 2.0, num=n_bins)
bins = [position_bins, velocity_bins, angle_bins, angular_velocity_bins]
def discretize(state):
    discretized_state = []
    for i in range(len(state)):
        discretized_state.append(np.digitize(state[i], bins[i]) - 1) # -1 to make the index start from 0
    return tuple(discretized_state)

# initialize gym environment
env = gym.make("CartPole-v1")
# env.metadata['render.modes'] = ['human']
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

q_table = np.zeros((n_bins,) * n_states + (n_actions,))
total_episodes = 50000
total_test_episodes = 100
max_steps = 99
alpha = 0.7  # Learning rate
gamma = 0.618
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = discretize(env.reset())
    step = 0
    done = False

    for step in range(max_steps):
        env.render()
        # Choose an action a in the current world state(s)
        action = epsilon_greedy_policy(state, epsilon)
        print(env.step(action))

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, _, _ = env.step(action)
        new_state = discretize(new_state)


        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        q_table[state][action] = update_q_table(state, action, reward, new_state, alpha, gamma)

        # Our new state is state
        state = new_state

        # If done : finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = discretize(env.reset())
    step = 0
    done = False
    total_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(q_table[state][:])

        new_state, reward, done, _, _ = env.step(action)
        new_state = discretize(new_state)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            print ("Score", total_rewards)
            break
        state = new_state
env.close()
print ("Score over time: " + str(sum(rewards) / total_test_episodes))
