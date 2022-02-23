import numpy as np
import matplotlib.pyplot as plt

n_trials = 1
alpha_thresh = 0.0001
c = -1
kappa = 0.75  #(0.0 - 0.9)  # recurrent inhibition strength
lambda_param = 1.38  #(0.0-0.9)  # lateral inhibition strength
tau = 0.5  # time constant of accumulator
recall_threshold = 1
noise_std = 0.1  # (0.0 - 0.8)
learning_rate = .2  # learning rate for SR


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
        self.x = np.zeros(self.n)
        self.rec = np.zeros((n_trials, self.n))

    def update(self, state, next_state):
        self.update_trace(state)
        self.update_sr(next_state, state)

    def update_sr(self, next_state, state):
        error = np.eye(self.n)[next_state] + self.gamma * self.M[next_state] - self.M[state]
        self.M += self.lr * np.outer(self.trace, error)  # takes two vectors to produce a NxM matrix

    def update_trace(self, state):
        # Vector addiction of trace vector and state vector 0,0,...1...0,0
        self.trace = self.gamma * self.decay * self.trace + np.eye(self.n)[state]

    def reset_trace(self):
        self.trace = np.zeros(self.n)

    def reset_matrix(self):
        self.M = np.zeros((self.n, self.n))

    def reset_x(self):
        self.x = np.zeros(self.n)

    def do_free_recall(self, recall_length):
        """Function to output list of recalled words.
        """
        ag.recalled_word_sequence = []
        accumulator_values = []
        ag.all_recall = []


        for w in recall_length:
            ag.x[ag.recalled_word_sequence] = 0

            ctx_vec = ag.trace  # contex vector  # TODO: figure out whether context vector should be set to zero
            f_strength = np.matmul(ctx_vec, ag.M)
            ag.all_retr[trial, :] = f_strength
            filtered = [x for x in f_strength if x > alpha_thresh]
            Cv = np.std(filtered) / np.mean(filtered)

            while True:
                self.update_accumulator(Cv, f_strength)
                #accumulator_values.append(ag.x) #moved to line 77
                if np.any(ag.x > recall_threshold):
                    crossed_thresh_idx = np.argwhere(ag.x > recall_threshold)
                    recalled_word = np.random.choice(crossed_thresh_idx[0])

                    ag.all_recall.append(recalled_word)

                    if recalled_word not in ag.recalled_word_sequence:
                        ag.recalled_word_sequence.append(recalled_word)
                        accumulator_values.append(ag.x)
                    ag.x[recalled_word] = 0  # TODO: if recalled word was recalled before, reset to zero but don't recall
                    break
            ag.update_trace(recalled_word)
        return ag.recalled_word_sequence, np.array(accumulator_values)

    def update_accumulator(self, coef_var, input_vector):
        """Update the accumulator of Sederberg et al.

        According to (Usher & McClelland, 2001, from Sederberg et al. 2008):
        xs(ag.x) = (1-tk-tdL)x_s0 + t*Cv*f_strength + e ; xs = max(xs,0)

        
        :param coef_var: coefficient of variation 
        :param input_vector:
        :return:
        """

        ag.x[ag.recalled_word_sequence] = 0
        noise = np.random.normal(0, noise_std, len(word_list))
        scaled_input = tau * coef_var * input_vector
        updated_state_vector = np.matmul(1 - tau * kappa - lambda_param * tau * L, ag.x)

        ag.x = updated_state_vector + scaled_input + noise
        ag.x = np.maximum(ag.x, 0)


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
    word_sequence = ['A', 'B', 'C', 'D', 'A', 'B', 'E', 'F', 'A', 'B', 'G', 'H', 'I', 'A']
    # translate to state indices:
    state_sequence = [word_index[word] for word in word_sequence]
    word_order = [word_index[word] for word in word_list]
    number_of_recalls = range(len(word_index) - 1)

    # initialise agent:
    ag = Agent(len(word_list), learning_rate=learning_rate)

    L = np.ones((len(word_list), len(word_list))) - np.eye(len(word_list))

    # learn for a given number of trials.
    for trial in range(n_trials):
        ag.reset_trace()
        ag.reset_matrix()
        ag.reset_x()
        # learning over M does not accumulate across different trials (each trial is tested with free recall
        # individually)
        for t in range(len(state_sequence) - 1):
            ag.update(state_sequence[t], state_sequence[t + 1])

        ag.SR_Ms[:, :, trial] = ag.M
        # ADD FREE RECALL AFTER EACH TRIAL HERE
        recalled_words, accum_vals = ag.do_free_recall(number_of_recalls)

        #plt.plot(accum_vals)
        #plt.legend(range(ag.n))
        #plt.show()
        print(recalled_words)


#print(ag.rec)





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
