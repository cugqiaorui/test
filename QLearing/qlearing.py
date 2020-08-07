import numpy as np
import pandas as pd
import time

np.random.seed(2)  # 计算机产生一组伪随机数列

N_STATES = 6  # the length of the 1 dimensional world 多少种状态，距离宝藏多少步
ACTIONS = ['left', 'right']  # available actions 两种可以选的状态
EPSILON = 0.9  # greedy police 百分之90选择最优的动作，百分之10选择随机的动作
ALPHA = 0.1  # learning rate
LAMBDA = 0.9  # discount factor 衰减因子，未来奖励的衰减值
MAX_EPISODES = 13  # maximum episodes 玩多少回合
FRESH_TIME = 0.3  # fresh time for one move 走一步花的时间


def build_q_table(n_states, actions):
    table = pd.DataFrame(  # 创建一个表格
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions,
    )
    # print(table)  # show table
    return table


def choose_action(state, q_table):
    # This is how to choose a action
    state_action = q_table.iloc[state, :]  # q表中状态为state时，所有对应的动作值赋值给state_action
    if (np.random.uniform() > EPSILON) or ((state_action == 0).all()):
        action_name = np.random.choice(ACTIONS)  # 随机选择
    else:
        action_name = state_action.idxmax()  # 百分之九十选择最大的state_action
    return action_name


# 创建环境以及环境对行为做出的feedback 反应

def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':  # move right
        if S == N_STATES - 2:  # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
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
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ_table:\n')
    print(q_table)
