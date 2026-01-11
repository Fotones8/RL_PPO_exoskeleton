#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

#Packages for ExoSim and RL algorithm
import gymnasium as gym
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.mean = nn.Linear(64, action_dim)

        # log_std as a learnable parameter (good enough for PPO)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, s):
        x = self.net(s)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)


class ExoSim(gym.Env):
    def __init__(self):
        print('Initiating environment')
        super().__init__()

        self.target = [1.5, -1.0]
        self.bones = [1.0, 1.0, 0.5]

        # Observation space: 3 joints angular position and target (x,y)
        self.observation_space = gym.spaces.Box(
            low=-3.0, high=3.0, shape=(8,), dtype=np.float32
        )

        # Action space: 3 joints position change
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

        self.state = None
        self.max_steps = 250
        self.steps = 0

    def reset(self, exoState, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset()
        # Start with a random 3-D vector
        
        self.state = np.array([float(exoState[0]), float(exoState[1]), float(exoState[2]),
                               float(exoState[3]), float(exoState[4]), float(exoState[5]), 
                               self.target[0], self.target[1]], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action, exoState):
        self.steps += 1   # increment step count
        
        # next_state
        self.state = np.array([float(exoState[0]), float(exoState[1]), float(exoState[2]),
                               float(exoState[3]), float(exoState[4]), float(exoState[5]), 
                               self.target[0], self.target[1]], dtype=np.float32)
        
        q0 = -self.state[0]
        q1 = self.state[1]
        q2 = -self.state[2]
        knee = [self.bones[0]*np.sin(q0), -self.bones[0]*np.cos(q0)]
        ankle = [knee[0]+self.bones[1]*np.sin(q0-q1), knee[1]-self.bones[1]*np.cos(q0-q1)]
        toes = [ankle[0]+self.bones[2]*np.sin(q0-q1+np.pi/2+q2), ankle[1]+self.bones[2]*np.cos(q0-q1+np.pi/2+q2)]

        # Reward: try to reach the target
        distance = [toes[0]-self.target[0], toes[1]-self.target[1]]
        reward = -abs(np.linalg.norm(distance))
        if abs(np.linalg.norm(distance)) < 0.1:
            reward = reward+1

        # Terminate when close enough to origin
        terminated = abs(np.linalg.norm(distance)) < 0.1
        truncated = self.steps >= self.max_steps

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print("State:", self.state)


#This class will create the node
class Node:
    def __init__(self):
        rospy.init_node('rl_node')
        self.pub1 = rospy.Publisher('/h3_sim/right_hip_effort_controller/command', Float64, queue_size=10)
        self.pub2 = rospy.Publisher('/h3_sim/right_knee_effort_controller/command', Float64, queue_size=10)
        self.pub3 = rospy.Publisher('/h3_sim/right_ankle_effort_controller/command', Float64, queue_size=10)
        self.pub4 = rospy.Publisher('/h3_sim/left_hip_effort_controller/command', Float64, queue_size=10)
        self.pub5 = rospy.Publisher('/h3_sim/left_knee_effort_controller/command', Float64, queue_size=10)
        self.pub6 = rospy.Publisher('/h3_sim/left_ankle_effort_controller/command', Float64, queue_size=10)
        
        self.subJointState = rospy.Subscriber('/h3_sim/joint_states', JointState, self.jointState_listener)
        print('All publisher and subscribers ready')
        
        self.env = ExoSim()
        self.exoState = []
        self.action_dim = 3
        self.state_dim = self.env.observation_space.shape[0]
        self.steps_per_update = 2048
        self.total_updates = 200
        self.current_frame = 0
        self.skip_frames = 3
        self.flag_ExoStateUpdated = False
        print('Env ready')

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        print('Neural networks ready')

        self.obs, _ = self.reset()
        self.ep_reward = 0
        self.rewards_array = []
        print('Setup ready!')

        for update in range(self.total_updates):
            self.policy_update(update)

    
    def jointState_listener(self, data):
        self.current_frame = self.current_frame + 1
        if self.current_frame % self.skip_frames == 0:
            position = data.position
            velocity = data.velocity
            self.exoState = [position[4],position[5],position[3],velocity[4],velocity[5],velocity[3]] 
            self.flag_ExoStateUpdated = True
        
    def step(self, action):
        # We publish the messages and wait for n frames to see the next state
        action = action.astype(np.float32)
        msg1 = Float64()
        msg2 = Float64()
        msg3 = Float64()
        msg1.data = float(action[0] * 15)
        msg2.data = float(action[1] * 10)
        msg3.data = float(action[2] * 10)
        self.pub1.publish(msg1)
        self.pub2.publish(msg2)
        self.pub3.publish(msg3)

        self.current_frame = 0
        self.flag_ExoStateUpdated = False
        while not self.flag_ExoStateUpdated:
            pass
        currentExoState = self.exoState
        # Now that we have the right exosState, we can continue
        return self.env.step(action=action, exoState=currentExoState)
    
    def reset(self):
        # We publish the messages and wait for n frames to see the next state
        # We publish -K*error to get back to 0 in every joint
        k = 20
        tolerance = 0.2 # The tolerance also introduces a bit of randomness in the reset, useful for the PPO
        flag_NotReset = True
        msg1 = Float64()
        msg2 = Float64()
        msg3 = Float64()
        while flag_NotReset:
            while not self.flag_ExoStateUpdated:
                pass
            
            flag_NotReset = False
            if abs(self.exoState[0]) > tolerance:
                flag_NotReset = True
                msg1.data = float(-self.exoState[0] * k)
                self.pub1.publish(msg1)
            if abs(self.exoState[1]) > tolerance:
                flag_NotReset = True
                msg2.data = float(-self.exoState[1] * k)
                self.pub2.publish(msg2)
            if abs(self.exoState[2]) > tolerance:
                flag_NotReset = True
                msg3.data = float(-self.exoState[2] * k)
                self.pub3.publish(msg3)
            
            self.current_frame = 0
            self.flag_ExoStateUpdated = False
        
        # All the joints are close to 0, we publish 0 in all torques
        msg1.data = float(0.0)
        msg2.data = float(0.0)
        msg3.data = float(0.0)
        self.pub1.publish(msg1)
        self.pub2.publish(msg2)
        self.pub3.publish(msg3)

        self.current_frame = 0
        self.flag_ExoStateUpdated = False
        while not self.flag_ExoStateUpdated:
            pass
        currentExoState = self.exoState
        # Now that we have the right exosState, we can continue
        return self.env.reset(exoState=currentExoState)

    def compute_advantage(self, rewards, values, dones, last_value, gamma=0.99, lamb=0.95):
        advantage = 0
        advantages = []
        returns = []
        values = values + [last_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma*values[t+1]*(1-dones[t]) - values[t]
            advantage = delta + gamma*lamb*(1-dones[t])*advantage
            advantages.insert(0, advantage)
        
        for t in range(len(advantages)):
            returns.append(advantages[t]+values[t])
        
        return (
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32)
        )

    
    def policy_update(self, update):
        # We prepare the batch results
        states, actions, logps, rewards, dones, values = [], [], [], [], [], []

        for step in range(self.steps_per_update):
            state = torch.tensor(self.obs, dtype=torch.float32)
            dist = self.actor(state)

            # We get the action and turn it into what the env expects
            raw_action = dist.sample()
            action = torch.tanh(raw_action) #From -1 to 1

            logp = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-7)
            lp = logp.sum(-1)

            # We get the value of the state and perform a step
            value = self.critic(state).item()

            #PUBLISH THE ACTIONS AND WAIT FOR THE RESULTS
            next_obs, reward, terminated, truncated, _ = self.step(action.numpy())
            done = terminated or truncated

            # We store in the batch all the results from this step
            states.append(self.obs)
            actions.append(raw_action.numpy())
            logps.append(lp.item())
            values.append(value)
            rewards.append(reward)
            dones.append(done)

            self.ep_reward += reward
            self.obs = next_obs

            #if done, we perform a reset
            if done:
                print('Episode reward: ', self.ep_reward)
                self.rewards_array.append(float(self.ep_reward))
                self.ep_reward = 0
                self.obs, _ = self.reset()
        
        # We prepare the batch by turning them into tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        logps = torch.tensor(logps, dtype=torch.float32)

        # We generate the last state value (predicting it)
        last_value = self.critic(torch.tensor(self.obs, dtype=torch.float32)).item()

        advantages, returns = self.compute_advantage(rewards, values, dones, last_value)

        # We update the ppo
        self.ppo_update(states, actions, logps, returns, advantages)

        print('~~~ FINISHED PPO UPDATE: ', update, ' ~~~')
        if update%10 == 0:
            print(self.rewards_array)

    def ppo_update(self, states, actions, old_logps, returns, advantages,
                   clip_eps=0.2, epochs=10, minibatch_size=64):
        n = len(states)
        indexes = np.arange(n) # We create an array from 0 to n-1
        # First we normalize the advantages to mean 0, std 1
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # We do several learning epochs over the same data
        for i in range(epochs):
            np.random.shuffle(indexes) #We randomise the order
            # We separe the indexes in minibatches
            for start in range(0, n, minibatch_size):
                minibatch = indexes[start:start+minibatch_size]
                # We extract the batch information
                mb_s = states[minibatch]
                mb_a = actions[minibatch]
                mb_old_lp = old_logps[minibatch]
                mb_returns = returns[minibatch]
                mb_adv = advantages[minibatch]
                # We compute the new log probabilities for the ratio
                dist = self.actor(mb_s)
                # We use a different logp to account for the tanh
                logp = dist.log_prob(mb_a) - torch.log(1 - torch.tanh(mb_a).pow(2) + 1e-7)
                lp = logp.sum(-1)

                # We calculate the ratio
                ratio = (lp-mb_old_lp).exp()
                # We do the clipping
                unclipped = ratio*mb_adv
                clipped = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)*mb_adv
                actor_loss = -torch.min(unclipped, clipped).mean()

                # We calculate the critic loss by predicting the values
                value_pred = self.critic(mb_s)
                critic_loss = F.mse_loss(value_pred.float(), mb_returns.float())

                # We update the actor
                self.opt_actor.zero_grad()
                actor_loss.backward()
                self.opt_actor.step()
                # We update the critic
                self.opt_critic.zero_grad()
                critic_loss.backward()
                self.opt_critic.step()
    
    #When the node starts
    def start(self):
        rospy.spin()


#This code will be executed when this ROS node is called with the command
if __name__ == '__main__':
    try:
        #We create the node and then start it
        node = Node()
        node.start()
        
    except rospy.ROSInterruptException:
        pass
