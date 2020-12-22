import gym
import random
import torch
import numpy as np
from collections import deque

## make env
env = gym.make('Breakout-v0')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

#make agent and test before train
from dqn_agent import Agent

agent = Agent(state_size= (210,160,3), action_size=4, seed=0)

# watch an untrained agent
state = env.reset()

score = 0
for j in range(200):
    state = state.transpose((2,0,1))
    action = agent.act(state)
    # env.render()
    state, reward, done, _ = env.step(action)
    score += reward
    if done:
        break 
print("untrained score:",score)
env.close()

## training 
def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state = state.transpose((2,0,1))
        score = 0

        while 1:
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.transpose((2,0,1))

            # print("in main:", np.shape(next_state))
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()
## outs:
print("scores in training process",scores)


## show:
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
scores = []
for i in range(100):
    state = env.reset()
    score = 0
    for j in range(200):
        state = state.transpose((2,0,1))
        action = agent.act(state)
        # env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break 
    scores.append(score)
env.close()

## outs:
print("ave scores in testing:",np.sum(scores)/len(scores),"scores in testing:",scores)
