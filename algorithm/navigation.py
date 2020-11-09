import gym
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(6)
robot = gym.make('RobotWorld-v0')

N_STATES = 25   # the length of the 1 dimensional world
# ACTIONS = ['left', 'right']     # available actions
ACTIONS = robot.getAction()
EPSILON = 0.7   # greedy police
ALPHA = 0.2     # learning rate
GAMMA = 1.0    # discount factor
MAX_EPISODES = 400   # maximum episodes
# FRESH_TIME = 0.3    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]

    # select
    select = 0

    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
        select = 0
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
        select = 1
    return action_name, select


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def ql():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = robot.reset()
        is_terminal = state in robot.terminate_states
        # print(is_terminal)
        # print("episode:%d\n" % (episode + 1))
        # print("%d -> " % state)
        # print("is_terminal: %s" % is_terminal)
        while not is_terminal:
            action, select = choose_action(state - 1, q_table)
            # key = '%d_%s'%(state, action)
            # print("%s -> \n" % action)

            # if select == 0:
            #     print("select randomly")
            # else:
            #     print("choose the best value")

            next_state, reward, is_terminal, info = robot.step(action)
            q_predict = q_table.loc[state-1, action]
            if not is_terminal:
                q_target = reward + GAMMA * q_table.iloc[next_state - 1, :].max()
            else:
                q_target = reward
            q_table.loc[state - 1, action] += ALPHA * (q_target - q_predict)
            step_counter += 1
            # print("%d -> %s" % (next_state, is_terminal))
    return q_table


def sarsa():
    sarsa_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        state = robot.reset()
        action, select = choose_action(state - 1, sarsa_table)
        while True:
            # state = robot.state
            next_state, reward, done, info = robot.step(action)
            next_action, select = choose_action(next_state - 1, sarsa_table)

            # q_predict = q_table.loc[state - 1, action]
            s_predict = sarsa_table.loc[state - 1, action]
            if not done:
                # q_target = reward + GAMMA * q_table.iloc[next_state - 1, :].max()
                s_target = reward + GAMMA * sarsa_table.loc[next_state - 1, next_action]
            else:
                s_target = reward
            # q_table.loc[state - 1, action] += ALPHA * (q_target - q_predict)
            sarsa_table.loc[state - 1, action] += ALPHA * (s_target - s_predict)
            state = next_state
            action = next_action
            if done:
                break

    return sarsa_table


if __name__ == "__main__":


    # sarsa
    # x_sarsa = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # y_sarsa = []
    # for x in x_sarsa:
    #     ALPHA = x
    #     sarsa_table = sarsa()
    #     sarsa_table['max_idx'] = sarsa_table.idxmax(axis=1)
    #     correct = 0
    #     #print("------test------\n")
    #     episode1 = 10
    #     for _ in range(episode1):
    #         # print("episode %d\n" % _)
    #         state1 = robot.reset()
    #         is_terminal = False
    #         # print("%d -> " % state1)
    #         while not is_terminal:
    #             # s_c = q_table.iloc[state1 - 1, :]
    #             # action_name = s_c.idxmax()
    #             action_name = sarsa_table.loc[state1 - 1, 'max_idx']
    #             # print("%s -> " % action_name)
    #             next_state, reward, is_terminal, info = robot.step(action_name)
    #             # print("%d -> " % next_state)
    #             state1 = next_state
    #         if state1 in robot.rewards:
    #             if robot.rewards[state1] == 1:
    #                 correct += 1
    #     accuracy = correct * 1.0 / episode1
    #     loss = 1 - accuracy
    #     y_sarsa.append(loss)
        # print("%f" % accuracy + "%")
    # plt.plot(x_sarsa, y_sarsa, label = 'sarsa')


    #plt.show()
    # print(sarsa_table)

    # print(sarsa_table)
    # # sarsa_table.to_csv('./sarsa1.csv')
    # y_q = []
    # for x in x_sarsa:
    #     ALPHA = x
    #     q_table = ql()
    #     q_table['max_idx'] = q_table.idxmax(axis=1)
    #     correct = 0
    #     #print("------test------\n")
    #     episode1 = 10
    #     for _ in range(episode1):
    #         # print("episode %d\n" % _)
    #         state1 = robot.reset()
    #         is_terminal = False
    #         # print("%d -> " % state1)
    #         while not is_terminal:
    #             # s_c = q_table.iloc[state1 - 1, :]
    #             # action_name = s_c.idxmax()
    #             action_name = q_table.loc[state1 - 1, 'max_idx']
    #             # print("%s -> " % action_name)
    #             next_state, reward, is_terminal, info = robot.step(action_name)
    #             # print("%d -> " % next_state)
    #             state1 = next_state
    #         if state1 in robot.rewards:
    #             if robot.rewards[state1] == 1:
    #                 correct += 1
    #     accuracy = correct * 1.0 / episode1
    #     loss = 1 - accuracy
    #     y_q.append(loss)
    # plt.plot(x_sarsa, y_q, label = 'Q')
    # plt.xlabel('learning rate')
    # plt.ylabel('loss')
    # plt.show()



    # sarsa_table = sarsa()
    # sarsa_table['max_idx'] = sarsa_table.idxmax(axis=1)
    # print(sarsa_table)
    # sarsa_table.to_csv('./sarsa_table.csv')
    #
    q_table = ql()
    # # actions = robot.getAction()
    # #add idmax
    q_table['max_idx'] = q_table.idxmax(axis=1)
    # print(q_table)
    # q_table.to_csv('./q_table.csv')


    for _ in range(50):
        print("episode %d" % _)
        state1 = robot.reset()
        is_terminal = False
        robot.render()
        print("%d -> " %state1)
        count = 0
        while not is_terminal:
            # action_name = np.random.choice(robot.getAction())
            action_name = q_table.loc[state1 - 1, 'max_idx']
            print("%s -> " % action_name)
            key = "%d_%s" % (state1, action_name)
            if key not in robot.t:
                while True:
                    action_name = np.random.choice(robot.getAction())
                    key = "%d_%s" % (state1, action_name)
                    if key in robot.t :
                        break
            next_state, reward, is_terminal, info = robot.step(action_name)
            print("%d -> " % next_state)
            state1 = next_state
            robot.render()
            time.sleep(0.5)
            count += 1
            if count > 15:
                break
        # time.sleep(1)






    # test
    # correct = 0
    # print("------test------\n")
    # episode1 = 10
    # for _ in range(episode1):
    #     print("episode %d\n" % _)
    #     state1 = robot.reset()
    #     is_terminal = False
    #     print("%d -> " % state1)
    #     while not is_terminal:
    #         # s_c = q_table.iloc[state1 - 1, :]
    #         # action_name = s_c.idxmax()
    #         action_name = sarsa_table.loc[state1 - 1, 'max_idx']
    #         print("%s -> " % action_name)
    #         next_state, reward, is_terminal, info = robot.step(action_name)
    #         print("%d -> " % next_state)
    #         state1 = next_state
    #     if state1 in robot.rewards:
    #         if robot.rewards[state1] == 1:
    #             correct += 1
    # accuracy = correct * 1.0 / episode1 * 100
    # print("%f" % accuracy + "%")
    # q_table.to_csv('./qlearning1.csv')







