import numpy as np

# 初始化参数
n_states = 16  # 状态数量，对应格子世界中的16个位置
n_actions = 4  # 行动数量，对应上、下、左、右四个方向的移动
q_table = np.zeros([n_states, n_actions])  # 初始化Q-table为全零
learning_rate = 0.1  # 学习率，决定了新信息对于旧信息的重视程度
discount_factor = 0.9  # 折扣因子，决定了未来奖励对当前决策的影响程度
epsilon = 0.1  # 探索率，决定了代理进行随机探索的概率
n_episodes = 10000  # 训练回合数

# Gridworld环境
gridworld = np.array([
    [0, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]
])
# 0表示普通格子，-1表示障碍，1表示目标位置

# 把2D坐标转化为1D状态
def coord_to_state(x, y, grid_shape):
    return y * grid_shape[0] + x

# 把1D状态转化为2D坐标
def state_to_coord(state, grid_shape): # grid_shape是个元组，grid_shape[0]是gridworld的行数，grid_shape[1]是gridworld的列数
    return state % grid_shape[0], state // grid_shape[0]

# 环境动态，state是gridworld当前2D坐标的对应状态，通过传入的action(分为上下左右4个运动方式移动)，计算出新的new_state,reward(奖励)，和done(完成标志)
def step(state, action):
    x, y = state_to_coord(state, gridworld.shape)

    if action == 0:  # 上
        y = max(y - 1, 0)
    elif action == 1:  # 下
        y = min(y + 1, gridworld.shape[0] - 1) #shape[0]的意思是总行数
    elif action == 2:  # 左
        x = max(x - 1, 0)
    elif action == 3:  # 右
        x = min(x + 1, gridworld.shape[1] - 1) #shape[1]的意思是总列数

    next_state = coord_to_state(x, y, gridworld.shape)  # 计算新的状态
    reward = gridworld[y, x]  # 获取奖励
    done = gridworld[y, x] == 1  # 检查是否到达目标位置

    return next_state, reward, done

# 开始训练
for i_episode in range(n_episodes):
    # 初始化状态
    state = 0  # 假设起始状态总是0，取值范围是0-15，代表gridworld中的16个位置
    done = False
    count = 0

    while not done:

        count = count + 1
        if (count) % 1000000 == 0:  # 每100个回合打印一次日志
            print("count is: {}".format(count))
            print(q_table)  # 打印最终的Q-table
        # 选择行动
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(n_actions)  # 探索：随机选择
        else:
            action = np.argmax(q_table[state])  # 利用：选择当前状态下最优的行动，np.argmax(arr)的作用是取arr中最大值对应的索引。所以这一句的意思，是根据Qtable,找到这一步对应的action

        # 执行行动并获取奖励和新的状态
        next_state, reward, done = step(state, action)

        # 更新Q-table
        old_value = q_table[state, action]  # 旧的Q值
        next_max = np.max(q_table[next_state])  # 新状态下的最大Q值

        # 通过贝尔曼方程进行Q值更新
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        q_table[state, action] = new_value  # 更新Q表

        state = next_state  # 更新状态

    # if i_episode<100:  # 每100个回合打印一次日志
    #     print("count is: {}".format(count))
    #     print(f"Episode {i_episode + 1}: done")
    #     print(q_table)  # 打印最终的Q-table

    if (i_episode + 1) % 100 == 0:  # 每100个回合打印一次日志
        print(f"Episode {i_episode + 1}: done")
        print(count)
        # print(q_table)

    # print(f"Episode {i_episode + 1}: done")
    # print(q_table)  # 打印最终的Q-table

print(q_table)  # 打印最终的Q-table