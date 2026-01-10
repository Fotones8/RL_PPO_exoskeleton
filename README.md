# RL_PPO_exoskeleton

This repository contains the implementation of a PPO algorithm for exoskeletons. It contains a simple PPO algorithm implementation for a simulator of an exoskeleton, and the necessary files to implement a PPO agent in the Exo-H3 exoskeleton simulator through ROS.

The present files are:

- **ppo_exo_simulator.ipynb**: a Jupyter notebook with the implementation of a simple PPO algorithm for a basic exoskeleton simulator

- **h3_sim_controllers2.launch**: torque controllers for the Exo-H3 exoskeleton simulator

- **rl_node.py**: ROS node implementing the PPO agent to control the Exo-H3 exoskeleton simulator through ROS.
