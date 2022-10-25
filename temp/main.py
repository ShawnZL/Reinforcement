import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9 # greedy police 90%选择最优解，10%选择随机解
ALPHA = 0.1 # learning rate 学习
GAMMA = 0.9 # discount factor
MAX_EPISODES = 20 # maximun episode 最多13回合，13回合已经可以训练可以
FRESH_TIME = 0.3 # fresh time for one move 0.3走一步

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    print(table)
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_,R

def update_env(S, episode, step_counter):
    # 更新环境函数，比如向右移动后，o表示的agent就距离宝藏更进一步，将agent所处位置实时打印
    env_list = ['_']*(N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' %(episode+1, step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r                     ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in MAX_EPISODES:
        step_counter = 0
        S = 0 # 每一个agent开始位置
        is_terminal = False
        update_env(S, episode, step_counter)
        while not is_terminal:


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table"\n')
    print(q_table)
