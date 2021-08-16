import numpy as np
import matplotlib.pyplot as plt
import random


class Agent(object):
    """Simple SR learning agent with eligibility traces.
    """
    def __init__(self, n_states, discount=0.95, decay=.8, learning_rate=.05):
        # parameters
        self.n = n_states
        self.gamma = discount  # this parameter controls the discount of future states.
        self.decay = decay  # this parameter controls the decay of the eligibility trace.
        self.lr = learning_rate

        # initialise
        self.M = np.zeros((self.n, self.n))  # the SR matrix
        self.trace = np.zeros(self.n)  # vector of eligibility traces

    def update(self, state, next_state):

        self.trace = self.gamma * self.decay * self.trace + np.eye(self.n)[state]

        error = np.eye(self.n)[next_state] + self.gamma * self.M[next_state] - self.M[state]
        self.M += self.lr * np.outer(self.trace, error)

    def reset_trace(self):
        self.trace = np.zeros(self.n)


if __name__ == '__main__':

    sequence_length = 100
    n_trials = 20

    # some example input words:
    text_body = 'The successor representation was introduced into reinforcement learning by Dayan'
    word_list = text_body.split()
    # associate each word with a unique state index:
    word_index = {}
    idx = 0
    for word in word_list:
        if word not in word_index:
            word_index[word] = idx
            idx += 1

    # generate some word sequence from the body of text (the word sequence the agent will see)
    word_sequence = [random.choice(word_list) for _ in range(sequence_length)]
    # translate to state indices:
    state_sequence = [word_index[word] for word in word_sequence]

    # initialise agent:
    ag = Agent(len(word_list))

    # learn for a given number of trials.
    for trial in range(n_trials):
        ag.reset_trace()
        for t in range(len(state_sequence) - 1):
            ag.update(state_sequence[t], state_sequence[t + 1])

    # plot SR matrix:
    plt.imshow(ag.M); plt.colorbar()
    plt.show()

