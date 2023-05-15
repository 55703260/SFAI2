import gym  # 导入gym库，这是一个游戏环境库
import torch  # 导入PyTorch库，这是一个用于深度学习的库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import numpy as np
import random
from collections import deque

# 定义深度Q网络（DQN）类
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 24)  # 定义第一层全连接层，输入节点数为4，输出节点数为24
        self.fc2 = nn.Linear(24, 48)  # 定义第二层全连接层，输入节点数为24，输出节点数为48
        self.fc3 = nn.Linear(48, 2)  # 定义第三层全连接层，输入节点数为48，输出节点数为2

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 第一层全连接层的输出经过ReLU激活函数
        x = torch.relu(self.fc2(x))  # 第二层全连接层的输出经过ReLU激活函数
        return self.fc3(x)  # 返回第三层全连接层的输出

# 定义获取动作的函数
def get_action(model, state, epsilon):
    # 如果随机数小于epsilon，则返回随机动作
    if np.random.rand() <= epsilon:
        return random.randrange(2)
    # 否则，返回模型预测的最优动作
    else:
        return torch.argmax(model(state)).item()

def train_model(model, target_model, optimizer, batch):
    # 从batch中提取状态、动作、奖励、下一个状态和结束标志
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.stack(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones)

    # 计算当前Q值和目标Q值
    current_q = model(states).gather(1, actions.unsqueeze(1))
    max_next_q = target_model(next_states).max(1)[0]
    expected_q = rewards + torch.logical_not(dones) * 0.99 * max_next_q

    # 计算损失并进行反向传播
    loss = torch.nn.functional.mse_loss(current_q.squeeze(), expected_q.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 主函数
def main():
    # 创建环境
    env = gym.make('CartPole-v1')
    # 创建模型和目标模型
    model = DQN()
    target_model = DQN()
    # 加载模型的参数到目标模型
    target_model.load_state_dict(model.state_dict())
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 创建经验回放缓冲
    replay_buffer = deque(maxlen=10000)
    # 设置初始的epsilon值
    epsilon = 1.0

    # 开始进行500轮游戏
    for i_episode in range(500):
        # 重置环境并获取初始状态
        state = env.reset()
        state = torch.tensor(state[0], dtype=torch.float32)

        # 在一轮游戏中进行最多201步操作
        for t in range(201):
            # 根据当前状态和epsilon值获取一个动作
            action = get_action(model, state, epsilon)

            # 执行动作并获取下一个状态、奖励和是否结束
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            # 如果游戏结束且没有完成任务，则奖励为-100
            reward = reward if not done or t == 200 else -100
            # 将经验数据添加到回放缓冲区
            replay_buffer.append((state, action, reward, next_state, done))
            # 更新当前状态
            state = next_state
            # 如果游戏结束，跳出循环
            if done:
                break
            # 如果回放缓冲区中的经验数量大于2000，开始训练模型
            if len(replay_buffer) > 2000:
                batch = random.sample(replay_buffer, 64)
                train_model(model, target_model, optimizer, batch)
                # 如果epsilon大于0.01，让epsilon逐渐减小
                if epsilon > 0.01:
                    epsilon *= 0.999

            if i_episode % 10 == 0:
                target_model.load_state_dict(model.state_dict())
        # 每10轮游戏更新一次目标模型
        if i_episode % 10 == 0:
            print('Episode {}\tAverage Score: {}'.format(i_episode, np.mean([b[2] for b in replay_buffer][-10:])))

if __name__ == '__main__':
    main()
