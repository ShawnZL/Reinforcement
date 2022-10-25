import numpy as np
import pandas as pd
import time

np.random.seed(2) # reproducible

N_STATES = 6 # the length of the 1 dimensional world, 距离最终点距离
ACTIONS = ['left', 'right'] # available actions
EPSILON = 0.9 # greedy police 90%选择最优解，10%选择随机解
ALPHA = 0.1 # learning rate 学习
GAMMA = 0.9 # discount factor
MAX_EPISODES = 13 # maximun episode 最多13回合，13回合已经可以训练可以
FRESH_TIME = 0.3 # fresh time for one move 0.3走一步

def build_q_table(n_states, actions):
    """
    构造Q表
    :param n_states: 状态
    :param actions: 行为
    :return: 返回当前Q表权值
    """
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), # inital values
        columns = actions,
    )
    print(table)
    return table

def choose_action(state, q_table):
    """
    选择动作
    :param state: 状态
    :param q_table: Q表
    :return: 选择的左右方向
    """
    state_actions = q_table.iloc[state, :] # 将现在agent观测到的状态所对应的q值取出来
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()): #当生成随机数大于EPSILON或者状态state所对应的q值全部为0，随机选择动作
        action_name = np.random.choice(ACTIONS)
    # 步数出现浮动原因是因为有90%选择最优 ，10%现在随机数
    else:
        action_name = state_actions.idxmax() # 选择state所对应的使q值最大的动作
    return action_name

def get_env_feedback(S,A):
    """
    选择动作后，还要根据现在的状态和动作获得下一个状态，并且返回奖励，奖励是由环境给出的，用来评价动作的好坏
    这里的设置的，只有在获得宝藏后才给予奖励，无论是左右，给的奖励都是零
    :param S: 状态
    :param A: 动作
    S_ Snext S的下一个动作
    :return: 返回选择的状态动作
    """
    if A == 'right': # move right
        if S == N_STATES - 2: # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else: # move left
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    # 更新环境函数，比如向右移动后，o表示的agent就距离宝藏更进一步，将agent所处位置实时打印
    env_list = ['_']*(N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' %(episode+1, step_counter);
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r                     ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)

def rl():
    """
    main part of RL loop
    :return: 跟新后的Q表
    """
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0 # 记录走了多少步
        S = 0 # 每个episode开始的时候将agent初始化在最开始的位置
        is_terminated = False
        update_env(S, episode, step_counter) # 打印的就是o-----T
        while not is_terminated: # 判断是否结束
            A = choose_action(S, q_table) #根据当前的状态选择动作
            S_, R = get_env_feedback(S, A) # 上一步已经获得1S对应动作a，接下来我们需要获得下一步的状态
            q_predict = q_table.loc[S,A] # q_predict估计值
            if S_ != 'terminal': #要判断一下，下一个时间点是不是已经取得宝藏
                q_target = R + GAMMA * q_table.iloc[S_,:].max() # q_target真实值
            else: # 得到宝藏后，得到的下一个状态不在q表中，q_target的计算也不同
                q_target = R # 回合终止，没有下一个状态
                is_terminated = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_ # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
        print('\r\nQ-table"\n')
        print(q_table)
    return q_table
if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table"\n')
    print(q_table)

