import numpy as np


class SarsaAgent(object):

    def __init__(self, num_actions, state_shape, alpha, lamda, discount, device):
        self.use_raw = False
        self.num_actions = num_actions
        self.theta = np.zeros(state_shape)
        self.alpha = alpha
        self.discount = discount
        self.lamda = lamda

    def Q(self):
        value = sum(self.theta)
        return value

    def choose_action(self, state, theta, n_actions, tile_coder):
        return np.argmax([self.Q(tile_coder.phi(state, action), theta) for action in range(n_actions)])

    def feed(self, trajectories):

        (state, action, reward, next_state, done) = tuple(trajectories)

    @staticmethod
    def step(state):

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