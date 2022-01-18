import numpy as np
import matplotlib.pyplot as plt

n_trials = 2000
alpha_tr = 0.07

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
        self.SR_Ms = np.zeros((self.n, self.n, n_trials))
        self.all_retr = np.zeros((n_trials, self.n))
        self.x  = np.zeros((0, self.n))

    def update(self, state, next_state):
        self.trace = self.gamma * self.decay * self.trace + np.eye(self.n)[state]
        # Vector addiction of trace vector and state vector 0,0,...1...0,0

        error = np.eye(self.n)[next_state] + self.gamma * self.M[next_state] - self.M[state]
        self.M += self.lr * np.outer(self.trace, error)  # takes two vectors to produce a NxM matrix

    def reset_trace(self):
        self.trace = np.zeros(self.n)

    def reset_matrix(self):
        self.M = np.zeros((self.n, self.n))


if __name__ == '__main__':

    # some example input words:
    text_body = 'A B C D E F G H I'
    word_list = text_body.split()
    sequence_length = len(word_list)
    # associate each word with a unique state index:
    word_index = {}
    idx = 0
    for word in word_list:
        if word not in word_index:
            word_index[word] = idx
            idx += 1

    # generate some word sequence from the body of text (the word sequence the agent will see)
    word_sequence: list[str] = ['A', 'B', 'C', 'D', 'A', 'B', 'E', 'F', 'A', 'B', 'G', 'H', 'I', 'A']
    # translate to state indices:
    state_sequence = [word_index[word] for word in word_sequence]

    # initialise agent:
    ag = Agent(len(word_list))

    # learn for a given number of trials.
    for trial in range(n_trials):
        ag.reset_trace()
        ag.reset_matrix()
        # learning over M does not accumulate across different trials (each trial is tested with free recall
        # individually)
        for t in range(len(state_sequence) - 1):
            ag.update(state_sequence[t], state_sequence[t + 1])
        ag.SR_Ms[:, :, trial] = ag.M

        # ADD FREE RECALL AFTER EACH TRIAL HERE
        ctx_vec = ag.trace #contex vector
        f_strength = np.matmul(ctx_vec, ag.M)
        ag.all_retr[trial,:] = f_strength
        filtered = [x for x in f_strength if np.abs(x) > alpha_tr]
        Cv = np.std(filtered)/np.mean(filtered)
        # function (Usher & McClelland, 2001, from Sederberg et al. 2008): xs(ag.x) = (1-tk-tdL)x_s0 + t*Cv*f_strength + e ; xs = max(xs,0)







    #plt.plot(ag.all_retr[0,:])
    #plt.show()

    #plt.imshow(ag.all_retr)
    #plt.colorbar()
    #plt.show()

    #print(ag.M)
    #plt.imshow(ag.M);
    #plt.colorbar()
    #plt.show()


    #mStr_IA = np.average(ag.SR_Ms[8, 0, :])
    #mStr_IB = np.average(ag.SR_Ms[8, 1, :])
    #mStr_AB = np.average(ag.SR_Ms[0, 1, :])

    #print(mStr_IA)
    #print(mStr_IB)
    #print(mStr_AB)

    # plot SR matrix:
    #plt.imshow(ag.M)
    #plt.colorbar()
    #plt.show()
