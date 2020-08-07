from Sarsa.maze_env import Maze
from Sarsa.RL_brain import SarsaTable

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset() #每一个回合环境给出的观测值，即红色点的位置信息

        # RL choose action based on observation基于观测值挑选动作
        action = RL.choose_action(str(observation))
        while True:
            # fresh env环境刷新
            env.render()

            # RL take action and get next observaion and reward
            observation_,reward,done = env.step(action)

            # RL choose action based on next observation_基于下一个观测者采取的action
            action_ = RL.choose_action(str(observation_))

            #RL learn from this transition
            RL.learn(str(observation),action,reward,str(observation_),action_)

            # swap observation
            observation = observation_
            action = action_
            # break while loop when end of this episode
            if done:
                break
    # end of time
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions = list(range(env.n_actions)))

    env.after(100,update)
    env.mainloop()
