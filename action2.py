import time
import random
import pydirectinput

# 定义按键的顺序
keys = ['up', 'down', 'left', 'right']

try:
    while True:  # 无限循环
        key = random.choice(keys)  # 随机选择一个键
        pydirectinput.keyDown(key)
        time.sleep(0.1)  # 按键间的延迟，可以根据需要调整
        pydirectinput.keyUp(key)
        time.sleep(0.5)  # 在下一次按键前的延迟，可以根据需要调整
except KeyboardInterrupt:
    print("程序已结束")
