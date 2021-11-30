import numpy as np

import torch
import torch.nn as nn

import random


class Memory(object):
    def __init__(self, buffer_size, batch_size):
        self.memory_size = buffer_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, legal_actions, done):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = tuple(state, action, reward, next_state, legal_actions, done)
        self.memory.append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))


class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class SarsaAgent(object):

    def __init__(self, num_actions, state_dim, alpha, lamda, discount, device):
        self.use_raw = False
        self.num_actions = num_actions
        self.theta = np.zeros(state_dim)
        self.alpha = alpha
        self.discount = discount
        self.lamda = lamda

        self.q_network = QNetwork(state_dim=state_dim, num_actions=num_actions)
        self.q_target_network = QNetwork(state_dim=state_dim, num_actions=num_actions)

        # self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

    def feed(self, trajectories):

        (state, action, reward, next_state, done) = tuple(trajectories)

        blabla = 1

    def step(self, state):

        q_values = self.predict_q_values(state)
        return np.random.choice(list(state['legal_actions'].keys()))

    def eval_step(self, state):

        # TODO: comprendre ceci


        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info