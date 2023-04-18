import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import seaborn as sns
import pandas as pd

alphas = [1, 0]  # np.linspace(0, 1, 11)
n_trials = 500
alpha_thresh = 0.0001
c = -1
kappa = 0.67  # (0.0 - 0.9)  # recurrent inhibition strength
lambda_param = 1.17  # (0.0-0.9)  # lateral inhibition strength
tau = 0.5  # time constant of accumulator
recall_threshold = 1
noise_std = 0.0003  # (0.0 - 0.8)
learning_rate = .1  # .1  # learning rate for SR
beta = 40
decay_time = 1000

mat = spio.loadmat('SemGr_L1.mat',
                   squeeze_me=True)
groupDist = mat['semDistGroups']

mat = spio.loadmat('allCosL1.mat', squeeze_me=True)
allCos = mat['allCos']

all_cos_distances = []

for i in range(allCos.shape[0]):
    for j in range(i, allCos.shape[1]):
        all_cos_distances.append(allCos[i, j])
all_cos_distances = np.sort(all_cos_distances)


def softmax(x, beta):
    """Compute the softmax function.
    :param x: Data
    :param beta: Inverse temperature parameter.
    :return:
    """
    x = np.array(x)
    return np.exp(beta * x) / sum(np.exp(beta * x))


cosDist = softmax(allCos, beta)


class Agent(object):
    """Simple SR learning agent with eligibility traces.
    """

    def __init__(self, n_states, discount=0.95, decay=.8, learning_rate=.05):  # .05):
        # parameters
        self.n = n_states
        self.gamma = discount  # this parameter controls the discount of future states.
        self.decay = decay  # this parameter controls the decay of the eligibility trace.
        self.lr = learning_rate

        # initialise
        self.Mcs = np.zeros((self.n, self.n))  # the SR matrix
        self.Msc = np.eye(self.n)
        self.S = cosDist  # the SR matrix with semantic distances
        self.Q = np.zeros((self.n, self.n))  # weighted SR matrix
        self.trace = np.zeros(self.n)  # vector of eligibility traces
        self.SR_Ms = np.zeros((self.n, self.n, n_trials))
        self.all_retr = np.zeros((n_trials, self.n))
        self.x = np.zeros(self.n)
        self.rec = np.zeros((n_trials, self.n))

    def update(self, state, next_state):
        self.trace = self.update_trace(state)
        self.Mcs = self.update_sr(next_state, state)

    def update_sr(self, next_state, state):
        error = np.eye(self.n)[next_state] + self.gamma * self.Mcs[next_state] - self.Mcs[state]
        return self.Mcs + self.lr * np.outer(self.trace, error)  # takes two vectors to produce a NxM matrix

    def compute_cin(self, state):
        # self.Msc = self.Mcs.transpose() #np.eye(self.n)
        self.Msc = np.eye(self.n)
        x = np.eye(self.n)[state]
        return self.Msc.dot(
            x)  # self.Msc.dot(np.eye(self.n)[state])  # self.Mcs.transpose() is Msc in Zhouetal manuscript

    def update_trace(self, state):
        # Vector addiction of trace vector and state vector 0,0,...1...0,0
        self.Cin = self.compute_cin(state)
        return self.gamma * self.decay * self.trace + self.Cin
        # self.trace = self.gamma * self.decay * self.trace + np.eye(self.n)[state]

    def decay_trace(self):
        self.trace = self.gamma * self.decay * self.trace

    def reset_trace(self):
        self.trace = np.zeros(self.n)

    def reset_matrix(self):
        self.Mcs = np.zeros((self.n, self.n))
        # self.Msc = np.eye(self.n)

    def reset_x(self):
        self.x = np.zeros(self.n)

    def _do_free_recall(self, recall_length, aph):
        """Function to output list of recalled words.
        """
        ag.recalled_word_sequence = []
        accumulator_values = []
        ag.all_recall = []
        ag.Q = aph * ag.M + (1 - aph) * ag.S

        for w in recall_length:

            ctx_vec = ag.trace  # contex vector  # TODO: figure out whether context vector should be set to zero
            f_strength = np.matmul(ctx_vec, ag.Q)
            ag.all_retr[trial, :] = f_strength
            filtered = [x for x in f_strength if x > alpha_thresh]
            Cv = np.std(filtered) / np.mean(filtered)

            while True:
                self.update_accumulator(Cv, f_strength)
                accumulator_values.append(self.x)  # moved to line 77
                if np.any(self.x > recall_threshold):
                    crossed_thresh_idx = np.argwhere(self.x > recall_threshold)
                    recalled_word = np.random.choice(crossed_thresh_idx[:, 0])

                    self.all_recall.append(recalled_word)

                    if recalled_word not in self.recalled_word_sequence:
                        self.recalled_word_sequence.append(recalled_word)
                        accumulator_values.append(self.x)
                    self.x[
                        recalled_word] = 0  # TODO: if recalled word was recalled before, reset to zero but don't recall
                    break
            self.update_trace(recalled_word)
        return self.recalled_word_sequence, np.array(accumulator_values)

    def do_free_recall_sampling(self, recall_length, aph):
        self.Q = aph * self.Mcs + (1 - aph) * self.S
        already_recalled = np.zeros(self.n, dtype=bool)

        recall_list = []
        for _ in range(recall_length):
            f_strength = np.matmul(self.trace, self.Q)
            f_strength[already_recalled] = 0
            if f_strength.sum() == 0:
                recall_prob = np.ones(len(f_strength))
                recall_prob[already_recalled] = 0
                recall_prob /= recall_prob.sum()
            else:
                recall_prob = f_strength / f_strength.sum()
            recalled_word = np.random.choice(np.arange(len(recall_prob)), p=recall_prob)
            self.trace = self.update_trace(recalled_word)
            already_recalled[recalled_word] = True
            recall_list.append(recalled_word)

            unique, counts = np.unique(recall_list, return_counts=True)
            if np.any(counts > 1):
                print('stop')
        return recall_list

    def _do_free_recall_LBA(self, recall_length, aph):

        scale = .2  # width of normal distribution that we draw slope from
        max_intercept = 1.
        threshold = 1.
        self.Q = aph * self.Mcs + (1 - aph) * self.S

        already_recalled = np.zeros(self.n, dtype=bool)

        recall_list = []

        for w in recall_length:
            ctx_vec = self.trace
            f_strength = np.matmul(ctx_vec, self.Q)

            slope = np.random.normal(f_strength, scale)
            if np.all(slope[~already_recalled] <= 0):
                break
            slope[slope < 0] = np.finfo(float).eps
            intercept = np.random.uniform(0, max_intercept, size=len(f_strength))

            time_to_threshold = (threshold - intercept) / slope
            time_to_threshold[already_recalled] = np.nan
            recalled_word = np.nanargmin(time_to_threshold)

            self.trace = self.update_trace(recalled_word)
            already_recalled[recalled_word] = True
            recall_list.append(recalled_word)

            unique, counts = np.unique(recall_list, return_counts=True)
            if np.any(counts > 1):
                print('break')

        return recall_list

    def update_accumulator(self, coef_var, input_vector):
        """Update the accumulator of Sederberg et al.

        According to (Usher & McClelland, 2001, from Sederberg et al. 2008):
        xs(ag.x) = (1-tk-tdL)x_s0 + t*Cv*f_strength + e ; xs = max(xs,0)


        :param coef_var: coefficient of variation
        :param input_vector:
        :return:
        """

        self.x[ag.recalled_word_sequence] = 0
        noise = np.random.normal(0, noise_std, len(word_list))
        scaled_input = tau * coef_var * input_vector
        updated_state_vector = np.matmul(1 - tau * kappa - lambda_param * tau * L, self.x)

        self.x = updated_state_vector + scaled_input + noise
        self.x = np.maximum(ag.x, 0)


if __name__ == '__main__':

    # some example input words:
    text_body = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 1 2 3 4'
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
    word_sequence = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4']

    # translate to state indices:
    state_sequence = [word_index[word] for word in word_sequence]
    word_order = [word_index[word] for word in word_list]
    number_of_recalls = len(word_index) - 10

    for a in range(len(alphas)):

        alpha = alphas[a]
        # initialise agent:
        ag = Agent(len(word_list), learning_rate=learning_rate, decay=.95)

        L = np.ones((len(word_list), len(word_list))) - np.eye(len(word_list))

        all_recalled_words = []
        # learn for a given number of trials.

        for trial in range(n_trials):
            ag.reset_trace()
            ag.reset_matrix()
            ag.reset_x()
            # learning over M does not accumulate across different trials (each trial is tested with free recall
            # individually)
            for t in range(len(state_sequence) - 1):
                ag.update(state_sequence[t], state_sequence[t + 1])
                # ag.decay_trace()

            ag.SR_Ms[:, :, trial] = ag.Mcs

            # Delay between learning and recall
          #  for t in range(decay_time):
           #              ag.decay_trace()

            ag.reset_trace()

            recalled_words = ag.do_free_recall_sampling(number_of_recalls, alpha)
            while len(recalled_words) < ag.n:
                recalled_words.append(np.nan)
            all_recalled_words.append(recalled_words)
            # plt.plot(accum_vals)
            # plt.legend(range(ag.n))
            # plt.show()
            # print(recalled_words)

        # print(ag.rec)

        # check distribution of next words given word 0
        histogram = np.zeros(ag.n)
        for i, recall in enumerate(all_recalled_words):
            if 0 in recall:
                position_word0 = np.argwhere(np.array(recall) == 0)[0][0]
                if position_word0 < len(word_index) - 2:
                    next_word = recall[position_word0 + 1]
                    if not np.isnan(next_word):
                        histogram[next_word] += 1

        # check distribution of first recalled words
        first_words = np.array(all_recalled_words)[:, 0]
        _, firstCount = np.unique(first_words, return_counts=True)
        print(firstCount)

        f = plt.figure(1)
        plt.plot(firstCount / sum(firstCount) * 100)
        plt.xlabel("Serial Position")
        plt.ylabel("First recall")
        # loc = np.array([0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59])  # range(len(lagCRP))
        # labels = [-29, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 29]  # list(range(-(len(state_sequence) - 1), len(state_sequence)))
        loc = np.array([0, 6, 11, 16, 21, 26, 30])  # range(len(lagCRP))
        labels = [1, 5, 10, 15, 20, 25, 29]  # list(range(-(len(state_sequence) - 1), len(state_sequence)))
        plt.xticks(loc, labels)  # choose which x locations to have ticks
        plt.title("Alpha =" + str(alpha))
        # plt.ylim([0.05, 0.15])
        f.show()
        title = "firstRecall_alpha" + str(alpha) + ".png"
        # f.savefig(title, dpi=f.dpi)
        # plt.close()

        # plot probability of recall

        all_words = np.array(all_recalled_words)
        all_words = all_words[~np.isnan(all_words)]
        _, allCount = np.unique(all_words, return_counts=True)
        print(allCount)

        g = plt.figure(2)
        plt.plot(allCount / n_trials)
        plt.xlabel("Serial Position")
        plt.ylabel("Recall")
        # loc = range(len(allCount))
        # labels = list(range(len(state_sequence)))
        loc = np.array([0, 6, 11, 16, 21, 26, 30])  # range(len(lagCRP))
        labels = [1, 5, 10, 15, 20, 25, 29]  # list(range(-(len(state_sequence) - 1), len(state_sequence)))
        plt.xticks(loc, labels)  # choose which x locations to have ticks
        plt.title("Alpha =" + str(alpha))
        plt.ylim([0, 1])
        g.show()
        title = "Recall_alpha" + str(alpha) + ".png"
        # g.savefig(title, dpi=g.dpi)
        # plt.close()

        # plot probability of recall given lag

        lagCRP = np.zeros((len(state_sequence) - 1) * 2 + 1)
        lagCRP_trial = np.zeros((n_trials, (len(state_sequence) - 1) * 2 + 1))
        lagCond_trial = np.zeros((n_trials, (len(state_sequence) - 1) * 2 + 1))
        cosDistGroups = np.zeros((n_trials, 4))
        distCont = np.zeros((n_trials, len(all_cos_distances)))
        lagCRPProb = np.zeros((n_trials, (len(state_sequence) - 1) * 2 + 1))

        for i, recall in enumerate(all_recalled_words):
            distGr = np.zeros(4)
            dCont = np.zeros(len(all_cos_distances))
            lagMatrix = np.zeros((ag.n, (len(state_sequence) - 1) * 2))
            lagM = np.zeros((ag.n, (len(state_sequence) - 1) * 2 + 1))

            for r in range(ag.n):
                lagMatrix[r, len(state_sequence) - 1 - r:len(state_sequence) - r + len(state_sequence) - 2] = np.ones(
                    (1, len(state_sequence) - 1))

            lagMatrix = np.insert(lagMatrix, len(state_sequence) - 1, [0], axis=1)

            for w, word in enumerate(recall):
                idxClue = [i for i, j in enumerate(state_sequence) if j == word]

                if len(idxClue) > 0:
                    lagM[w, :] = lagMatrix[idxClue[0], :]
                    for lt in range(ag.n):
                        lagMatrix[lt, idxClue[0] - lt + len(state_sequence) - 1] = 0

                if w < len(word_index) - 2:
                    next_word = recall[w + 1]
                    idxRetr = [i for i, j in enumerate(state_sequence) if j == next_word]

                    if not np.isnan(next_word):
                        diffLag = [x - y for x in idxRetr for y in idxClue]
                        lag = min(diffLag, key=abs)
                        lagCRP[lag + len(state_sequence) - 1] += 1
                        lagCRP_trial[i, lag + len(state_sequence) - 1] += 1
                        dGroup = groupDist[idxRetr, idxClue]
                        distGr[dGroup - 1] += 1
                        distance = allCos[idxRetr, idxClue]
                        cosInd = np.where(all_cos_distances == distance)
                        dCont[cosInd] += 1

            cosDistGroups[i, :] = distGr
            distCont[i, :] = dCont
            lagCRPProb[i, :] = np.sum(lagM, axis=0)
        distCont = np.sum(distCont, 0) / len(all_cos_distances) * 100
        # print(distCont)

        lagCRP_method = np.zeros((n_trials, (len(state_sequence) - 1) * 2 + 1))
        for c in range(n_trials):
            actCount = lagCRP_trial[c, :]
            possCount = lagCRPProb[c, :]
            CRP = np.divide(actCount, possCount)
            CRP[np.isnan(CRP)] = 0
            lagCRP_method[c, :] = CRP

        lagCRP_final = np.nanmean(lagCRP_method, axis=0)

        d = plt.figure(3)
        plt.plot(lagCRP_final)
        plt.xlabel("Lag")
        plt.ylabel("Cond. Resp. Probability")
        loc = np.array([0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59])  # range(len(lagCRP))
        labels = [-29, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25,
                  29]  # list(range(-(len(state_sequence) - 1), len(state_sequence)))
        plt.xticks(loc, labels)  # choose which x locations to have ticks
        plt.title("Alpha =" + str(alpha))
        # plt.ylim([0, 1.5])
        d.show()
        title = "CRP_alpha_method" + str(alpha) + ".png"
        # d.savefig(title, dpi=d.dpi)
        # plt.close()

        # d = plt.figure(3)
        # plt.plot(lagCRP / n_trials)
        # plt.xlabel("Lag")
        # plt.ylabel("Cond. Resp. Probability")
        # loc = np.array([0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59])  # range(len(lagCRP))
        # labels = [-29, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25,
        #         29]  # list(range(-(len(state_sequence) - 1), len(state_sequence)))
        # plt.xticks(loc, labels)  # choose which x locations to have ticks
        # plt.title("Alpha =" + str(alpha))
        # plt.ylim([0, 1.5])
        # d.show()
        # title = "CRP_alpha" + str(alpha) + ".png"
        # d.savefig(title, dpi=d.dpi)
        # plt.close()

        # plot normalised CRP graph for number of words in each lag
        # lab = list(range(-(len(state_sequence) - 1), len(state_sequence)))
        # distCRP = np.array(lab)
        # nCRP = np.zeros(len(distCRP))
        # nCRP = nCRP[0]
        # for c, nw_CRP in enumerate(distCRP):
        #   crp = len(state_sequence) - abs(nw_CRP)
        #  nCRP[c] = crp
        # nCRP[len(state_sequence) - 1] = 0

        # e = plt.figure(4)
        # normCRP = 100 * lagCRP_final / (n_trials * nCRP)
        # plt.plot(normCRP)
        # plt.xlabel("Lag")
        # plt.ylabel("Normalised Cond. Resp. Probability %")
        # loc = range(len(lagCRP))
        # labels = list(range(-(len(state_sequence) - 1), len(state_sequence)))
        # plt.xticks(loc, labels)  # choose which x locations to have ticks
        # plt.title("Alpha =" + str(alpha))
        # .ylim([0, 8])
        # e.show()
        # title = "Norm_CRP_alpha" + str(alpha) + ".png"
        # e.savefig(title, dpi=d.dpi)
        # plt.close()

        df = cosDistGroups / len(recall) * 100
        f = plt.figure(5)
        ax = f.add_subplot(111)
        # bd = ax.add_axes([0, 0, 1, 1])
        bp = ax.boxplot(df)
        plt.title("Alpha =" + str(alpha))
        ax.set_xticklabels(['small', 'medium-small',
                            'medium-large', 'large'])
        plt.xlabel("Cosine Distance Word2Vec")
        plt.ylabel('Conditional Response Probability %')
        f.show()
        title = "cosDistGroup_alpha" + str(alpha) + ".png"
        # f.savefig(title, dpi=d.dpi)
        # plt.close()

        # y = plt.figure(6)
        # plt.scatter(all_cos_distances, distCont)
        # # calculate equation for trendline
        # z = np.polyfit( all_cos_distances, distCont, 1)
        # p = np.poly1d(z)
        # # add trendline to plot
        # plt.plot(all_cos_distances, p(all_cos_distances))
        # plt.title("Alpha =" + str(alpha))
        # plt.xlabel("Cosine Distance Word2Vec")
        # plt.ylabel('Conditional Response Probability %')
        # y.show()
        # title = "cosDistCont_alpha" + str(alpha) + ".png"
        # #y.savefig(title, dpi=d.dpi)
        # plt.close()

        features = ["CosDist", "recContDist"]
        df = pd.DataFrame(columns=features)
        df['CosDist'] = all_cos_distances
        df['recContDist'] = distCont
        y = plt.figure(6)
        sns.regplot(x="CosDist", y="recContDist", data=df, order=2)
        plt.title("Alpha =" + str(alpha))
        plt.xlabel("Cosine Distance Word2Vec")
        plt.ylabel('Conditional Response Probability %')
        y.show()
        title = "cosDistCont_alpha" + str(alpha) + ".png"
    # y.savefig(title, dpi=d.dpi)
    # plt.close()

    # plt.plot(ag.all_retr[0,:])
    # plt.show()

    # plt.imshow(ag.all_retr)
    # plt.colorbar()
    # plt.show()

    # print(ag.M)
    # plt.imshow(ag.M);
    # plt.colorbar()
    # plt.show()

    # mStr_IA = np.average(ag.SR_Ms[8, 0, :])
    # mStr_IB = np.average(ag.SR_Ms[8, 1, :])
    # mStr_AB = np.average(ag.SR_Ms[0, 1, :])

    # print(mStr_IA)
    # print(mStr_IB)
    # print(mStr_AB)

    # plot SR matrix:
    # plt.imshow(ag.M)
    # plt.colorbar()
    # plt.show()
