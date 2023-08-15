# Shadow-Hand Controller

We constructed a controller for Shadow-Hand model in Mujoco environment using Deep Learning & Deep Reinforcement Learning. The controller allows the hand to perform sign-language gestures.

# Shadow-Hand Description

![shadow-hand-image](https://github.com/deepmind/mujoco_menagerie/blob/main/shadow_hand/shadow_hand.png)

Shadow-Hand is a 3D Robotic hand provided by *mujoco_menagerie* repository, for academic and research purposes. It can be found here: https://github.com/deepmind/mujoco_menagerie/tree/main/shadow_hand

# How it works

Shadow-Hand uses 20 Positional Motors as actuators to enable movement on its fingers and its wrist. It's actuator has a limited control range defined by the manufacturer. The control range of this hand are provided below:

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

# Python Version & Libraries
* **python==3.9** https://www.python.org/downloads/release/python-390/
* **mujoco=2.3.7** https://github.com/deepmind/mujoco
* **tensorflow==2.9.1** https://www.tensorflow.org/install
* **ray[rllib]==2.3.1** https://docs.ray.io/en/latest/rllib/index.html
* **gymnasium==0.26.1** https://gymnasium.farama.org/
* **matplotlib==3.7.2** https://matplotlib.org/
