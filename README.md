# DeepRL-ND-Continuous-Control

## Project 2: Continuous Control

### Introduction

For this project, Work with the Reacher environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible. For multi-agnets it contains 20 identical agents, each with its own copy of the environment.

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.

- This yields an average score for each episode (where the average is over all 20 agents).
- The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). 

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. In the case of the plot above, the environment was solved at episode 63, since the average of the average scores from episodes 64 to 163 (inclusive) was greater than +30.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

2. If you haven't setup Unity openAI gym environment, please follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment.


## Instructions

```
git clone https://github.com/bmaxdk//DeepRL-ND-Continuous-Control.git
cd DeepRL-ND-Continuous-Control
```

### Select Option for Algorithms:
#### (Option DDPG)
* Follow the instructions in [`Continuous_Control_final.ipynb`](https://github.com/bmaxdk/DeepRL-ND-Continuous-Control/blob/main/DDPG/Continuous_Control_final.ipynb) to train and run the agent!
```
cd DDPG
```
Use jupyter notebook to open [`Continuous_Control_final.ipynb`](https://github.com/bmaxdk/DeepRL-ND-Continuous-Control/blob/main/DDPG/Continuous_Control_final.ipynb)


#### (Option D4PG)
* Follow the instructions in [`Continuous_Control_final.ipynb`](https://github.com/bmaxdk/DeepRL-ND-Continuous-Control/blob/main/D4PG/Continuous_Control_final.ipynb) to train and run the agent!
```
cd D4PG
```
Use jupyter notebook to open [`Continuous_Control_final.ipynb`](https://github.com/bmaxdk/DeepRL-ND-Continuous-Control/blob/main/D4PG/Continuous_Control_final.ipynb)
