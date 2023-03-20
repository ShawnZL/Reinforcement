import numpy as np
import pandas as pd
import time

np.random.seed(2) # reproducible  the same number

# env
N_STATE = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1 # learning rate
LAMBDA = 0.9 # discount factor
MAX_EPISODES = 13 # max episodes
FRESH_TIME = 0.3 # fresh time for one move

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions, #actions' name   columns's lable
    )
    print(table)
    return table

def choose_action(state, q_table):
    # this is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else: # act greedy
        action_name = state_actions.argmax() # choose the max one
    return action_name

def get_env_feedback(S, A):
    # this is how agent will interact with the environment
    if A == 'right':
        if S == N_STATE - 2: # terminate N_STATE - 1为end，然后又向右走
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S # reach the wall
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATE-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    # main part of the RL loop
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminal = False
        update_env(S, episode, step_counter)
        while not is_terminal:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A) # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_,:].max() # next state is not terminal
            else:
                q_target = R # the end
                is_terminal = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/