# Shadow-Hand Controller

We constructed a controller for Shadow-Hand model in Mujoco environment using Deep Learning & Deep Reinforcement Learning. The controller allows the hand to perform sign-language gestures. The supported gestures of this hand are:

1. Rest
2. Drop
3. Middle Finger
4. Yes
5. No
6. Rock
7. Circle

# Shadow-Hand Description

![shadow-hand-image](https://github.com/deepmind/mujoco_menagerie/blob/main/shadow_hand/shadow_hand.png)

Shadow-Hand is a 3D Robotic hand provided by *mujoco_menagerie* repository, for academic and research purposes. It can be found here: https://github.com/deepmind/mujoco_menagerie/tree/main/shadow_hand

# How it works

Shadow-Hand uses 20 Positional Motors as actuators to enable movement on its fingers and its wrist. It's actuator has a limited control range defined by the manufacturer. The positions of each actuator can be found by downloading the Mujoco Simulator and importing the Shadow-Hand XML file (located in *objecs/shadow_hand/scene_left.xml*) into the simulator via **Drag & Drop**. To view the position and orientation of each actuator, *Joints* option must be enabled inside the simulation window, which can be found in Rendering/Model-Element panel. Alrnatively, they are analytically described inside the xml file.

The position as well as the control range of each actuator of the hand are provided below:

| id |	ctrl_limit_left	| ctrl_limit_right |
| ---|----------------- | ---------------- |
| 0 |	-0.523599	| 0.174533 |
| 1 |	-0.698132	| 0.488692 |
| 2 |	-1.0472	| 1.0472 |
| 3 |	0	| 1.22173 |
| 4 |	-0.20944	| 0.20944 |
| 5 |	-0.698132	| 0.698132 |
| 6 | 	-0.261799	| 1.5708 |
| 7 |	-0.349066	| 0.349066 |
| 8 |	-0.261799	| 1.5708 |
| 9 |	0	| 3.1415 |
| 10 |	-0.349066	| 0.349066 |
| 11 |	-0.261799	| 1.5708 |
| 12 |	0	| 3.1415 |
| 13 |	-0.349066	| 0.349066 |
| 14 |	-0.261799	| 1.5708 |
| 15 |	0	| 3.1415 |
| 16 |	0	| 0.785398 |
| 17 |	-0.349066	| 0.349066
| 18 |	-0.261799	| 1.5708 |
| 19 |	0	| 3.1415 |

![Shadow-Hand-Joints](https://github.com/kochlisGit/Shadow-Hand-Controller/blob/main/shadow-hand-joints.png)

# Behavioral Cloning

Behavioral Cloning (BC) is a method of teaching controllers to perform tasks by observing and imitating human experts. It's particularly common technique in robotics, where a model learns to perform tasks by imitating humans. This method involves:

1. **Data Collection**: A human expert should construct a dataset, which consists of pairs: **(Observations, Actions)**.
2. **Learning Algorithm**: A learning algorithm is then designed in order to map the Observations (Inputs) to expected Actions (Outputs).
3. **Deployment**: The controller is then evaluated & deployed on the physical model.

# Deep Reinforcement Learning

Deep Reinforcement Learning (DRL) is another popular technique for teaching machines to perform tasks in an optimal way, but it differs from Behavioral Cloning. While Behavioral Cloning learns directly from examples (demonstrations) of desired behavior, DRL learns through interaction with an environment and receiving feedback in the form of rewards or penalties. In DRL environments, an agent receives its state $s_{t}$ from the environment, chooses an action $a_{t}$ using its policy $\pi_{\theta}$, then transits into the next state $s_{t+1}$ and finally receives a reward $r_{t+1}$ based on how good the action was. The goal of the agent is to maximize the cumulative returns $R = r_{t+1} + r_{t+2} + r_{t+3} + ...$. 

There are two popular family of algorithms that are used in DRL problems:
1. **Value-Based Methods**: These methods aim to find a value function, which is a measure of the expected cumulative reward an agent can obtain from any given state (or state-action pair). The most common value function is the Q-function, Q(s,a), which measures the expected return from taking action $a_{t}$ in a state $s_{t}$. Another way of measuring the expected return is via the use of $V(s)$, which measures how good is a state $s_{t}$. Sometimes, these two functions are combined in order to speedup the training process and learning performance of the agent. By estimating the value function correctly, the agent can make optimal transition between consecutive states, and thus learn an optimal behavior. Both Q-Function and Value functions can be estimated using a Neural Network, as shown below.
 
![](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*wfKvMsVMkUhEGz1YH7kCQA.png)

2. **Policy-Gradient Methods**: Instead of computing the estimated returns, Policy-Gradient methods directly optimize the policy function without the necessity of a value function as an intermediary. The policy is typically parameterized by a set of weights (e.g. a neural network), and learning involves adjusting these weights to maximize expected reward. The idea is that by changing the weights of a neural network, and thus changing the selected actions, it also changes the rewards that the agent receives by the environment. The goal of the agent is to changes its policy (its weights) in the directin that maximizes its rewards, as shown below. One major advantage of Policy-Gradient methods over Value-Based methods is that they allow the use of continuous actions (e.g. float values), instead of discrete actions (e.g. 1,2,3). This is specifically useful in Mujoco environments, such as ours, where the goal is to predict the control (float values) of each actuator.

![](https://assets.website-files.com/62699d12d5bdec228b8eb739/62699d12d5bdec222e8eb9b6_61e98062f113c14630a6a066_hzUAOMVlW5j8lXhv1mSbwpU5ihOBEwnxoRd4twRoiy17ZgYx09VJzuxz_TQvkHkS8NinIkWGsIjloVHo3tPyRt14lZ-C2HszlcT1YjDDNP8kZ6hqzkVg5pf_t1VHTw.png)

# Observations (Inputs)

Both Behavioral Cloning & DRL technique require a dataset or a simulation environment to retrieve the data. To train both agents, we constructed a dataset which consists of pairs $(Sign, Order) --> (Control)$.

* Sign: Sign is a set of sequential controls, which the hand controller has to execute, in order to perform a sign gesture. For example, in order to perform a "Yes" gesture, the hand must first make a fist (1), then move its hand down (2) and then up again (3).
* Order: Order is the index of the desired control inside the sequence. This is a necessary, because the agent should execute all controls sequentially, in a specified order. For example, in the "Yes" example, the controls (1), (2), (3) must be executed as consecutively.
* Control: Control is a vector (array) of 20 values, one for each actuator. Each value is a float number that controls the position of the actuator and does not exceed its control limits described the above table.

# Sign/Order Representation

The neural network receives an input vector and outputs the control vector. While the input vector is expected to be a vector of float values, our dataset contains Signs, which are strings (words) and Orders, which are integer numbers. Because the dataset is very small, we converted each word and order to a unique vector as shown below:

| sign |	    vector	   |
|------|-----------------|
| rest | [0,0,0,0,0,0,1] |
| drop | [0,0,0,0,0,1,0] |
| middle finger | [0,0,0,0,1,0,0] |
| yes | [0,0,0,1,0,0,0] |
| no | [0,0,1,0,0,0,0] |
| rock | [0,1,0,0,0,0,0] |
| circle | [1,0,0,0,0,0,0] |

| order |	    vector	   |
|------|-----------------|
| 1 | [0,0,1] |
| 2 | [0,1,0] |
| 3 | [1,0,0] |

Now, these features can be concatenated and inserted into the neural network/DRL agent controller.

# Neural Network (BC)

The goal of the neural network is to predict the control values of the 20 actuators: $ \hat{y_{0}}, \hat{y_{1}}, \hat{y_{1}}, ..., \hat{y_{19}} $, by using the sign,order pair as inputs. To do that, the network outputs a control prediction, and uses the Mean Absolute Error (MAE) function in order to evaluate its prediction error. Then, the network uses *Adam* optimizer, which is an improvement of *Gradient Descent* algorithm, in order to update its weights and reduce the MAE. If $y$ and $\hat{y}$ are the target (actual) and predicted control respectively, then MAE is defined as:

$\frac{1}{N} * \sum_{i=1}^{N} |y_i - \hat{y_i}|$

![](https://github.com/kochlisGit/Shadow-Hand-Controller/blob/main/figures/nn_performance.png)

# Proximal Policy Optimization Peformance (DRL)

Proximal Policy Optimization (PPO) is a popular policy gradient algoithm that addresses some challenges in training stability and efficiency faced by Trust Region Policy Optimization (TRPO). PPO introduces a clipping mechanism to prevent the policy from being updated too drastically in any single step (prevents the weights from receiving large updates and change drastically), ensuring that the new policy doesn't deviate too much from the old one. When udating the policy, instead of directly maximizing the expected reward, PPO aims to maximize a clipped version of the objective function. This clipped objective limits the ratio of the probabilities of the new and old policies. Specifically, if the new policy would increase the action's probability significantly compared to the old policy, this change is clipped to be within a specified range (e.g., between 0.8 and 1.2). This prevents overly aggressive updates which could quickly converge into sub-optimal policies. This range is defined by a clipping parameter $e$, which is typically set between $\[0.1, 0.3\]$.

Just like BC Neural Network, PPO receives pairs of (sign, order) as inputs and outputs the target control of the hand. Then, instead of using a Loss (Error) Function to evaluate its error, it uses a reward function to receive rewards, which it tries to maximize in each iteration. The reward function is defined as $r_{t} = \frac{1}{euclidean(y_{t} - \hat{y_{t}})}$, where *euclidean* is the euclidean distance between the predicted and target control. 

![](https://github.com/kochlisGit/Shadow-Hand-Controller/blob/main/figures/ppo_performance.png)

# Python Version & Libraries
* **python==3.9** https://www.python.org/downloads/release/python-390/
* **mujoco=2.3.7** https://github.com/deepmind/mujoco
* **tensorflow==2.9.1** https://www.tensorflow.org/install
* **ray[rllib]==2.3.1** https://docs.ray.io/en/latest/rllib/index.html
* **gymnasium==0.26.1** https://gymnasium.farama.org/
* **matplotlib==3.7.2** https://matplotlib.org/
