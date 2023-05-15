import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def get_action(model, state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(2)
    else:
        return torch.argmax(model(state)).item()

def train_model(model, target_model, optimizer, batch):
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.stack(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones)

    current_q = model(states).gather(1, actions.unsqueeze(1))
    max_next_q = target_model(next_states).max(1)[0]
    expected_q = rewards + torch.logical_not(dones) * 0.99 * max_next_q

    loss = torch.nn.functional.mse_loss(current_q.squeeze(), expected_q.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    model = DQN()
    target_model = DQN()
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    replay_buffer = deque(maxlen=10000)
    epsilon = 1.0

    for i_episode in range(500):
        state = env.reset()
        state = torch.tensor(state[0], dtype=torch.float32)
        for t in range(201):
            action = get_action(model, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = reward if not done or t == 200 else -100
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                break

            if len(replay_buffer) > 2000:
                batch = random.sample(replay_buffer, 64)
                train_model(model, target_model, optimizer, batch)
                if epsilon > 0.01:
                    epsilon *= 0.999

            if i_episode % 10 == 0:
                target_model.load_state_dict(model.state_dict())

        if i_episode % 10 == 0:
            print('Episode {}\tAverage Score: {}'.format(i_episode, np.mean([b[2] for b in replay_buffer][-10:])))

if __name__ == '__main__':
    main()
