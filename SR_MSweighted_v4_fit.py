import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import seaborn as sns
import pandas as pd
import nltk
import gensim as gen
#from nltk.corpus import wordnet as wn
import random
import gensim.downloader as api
from gensim.models import KeyedVectors
import scipy


# list of noun taken from https://memory.psych.upenn.edu/Word_Pools iEEG pool of common words
common_nouns = [
    "ANT", "APE", "ARK", "ARM", "AXE", "BADGE", "BAG", "BALL", "BAND", "BANK",
    "BARN", "BAT", "BATH", "BEACH", "BEAK", "BEAN", "BEAR", "BED", "BEE", "BELL",
    "BENCH", "BIRD", "BLOOM", "BLUSH", "BOARD", "BOAT", "BOMB", "BOOK", "BOOT",
    "BOWL", "BOX", "BOY", "BRANCH", "BREAD", "BRICK", "BRIDGE", "BROOM", "BRUSH",
    "BUSH", "CAGE", "CAKE", "CALF", "CANE", "CAPE", "CAR", "CARD", "CART", "CASH",
    "CAT", "CAVE", "CHAIR", "CHALK", "CHEEK", "CHIEF", "CHIN", "CLAY", "CLIFF",
    "CLOCK", "CLOTH", "CLOUD", "CLOWN", "COAT", "COIN", "CONE", "CORD", "CORN",
    "COUCH", "COW", "CRANE", "CROW", "CROWN", "CUBE", "CUP", "DAD", "DART", "DEER",
    "DESK", "DIME", "DITCH", "DOCK", "DOG", "DOLL", "DOOR", "DRESS", "DRUM",
    "DUCK", "EAR", "EEL", "EGG", "ELF", "FACE", "FAN", "FARM", "FENCE", "FILM",
    "FISH", "FLAG", "FLAME", "FLEA", "FLOOR", "FLUTE", "FOAM", "FOG", "FOOD",
    "FOOT", "FORK", "FORT", "FOX", "FROG", "FRUIT", "FUDGE", "FUR", "GATE",
    "GEESE", "GIRL", "GLASS", "GLOVE", "GOAT", "GOLD", "GRAPE", "GRASS", "GUARD",
    "HAND", "HAT", "HAWK", "HEART", "HEN", "HILL", "HOLE", "HOOF", "HOOK",
    "HORN", "HORSE", "HOSE", "HOUSE", "ICE", "INK", "JAIL", "JAR", "JEEP",
    "JET", "JUDGE", "JUICE", "KEY", "KING", "KITE", "LAKE", "LAMB", "LAMP",
    "LAND", "LAWN", "LEAF", "LEG", "LIP", "LOCK", "MAIL", "MAP", "MAT", "MAZE",
    "MILK", "MOLE", "MOON", "MOOSE", "MOTH", "MOUSE", "MOUTH", "MUD", "MUG",
    "MULE", "NAIL", "NEST", "NET", "NOSE", "OAK", "OAR", "OWL", "PALM",
    "PANTS", "PARK", "PASTE", "PEA", "PEACH", "PEAR", "PEARL", "PEN", "PET",
    "PHONE", "PIE", "PIG", "PIN", "PIPE", "PIT", "PLANE", "PLANT", "PLATE",
    "POLE", "POND", "POOL", "PRINCE", "PURSE", "QUEEN", "RAIN", "RAKE", "RAT",
    "RIB", "RICE", "ROAD", "ROCK", "ROOF", "ROOM", "ROOT", "ROPE", "ROSE",
    "RUG", "SAIL", "SALT", "SCHOOL", "SEA", "SEAL", "SEAT", "SEED", "SHARK",
    "SHEEP", "SHEET", "SHELL", "SHIELD", "SHIP", "SHIRT", "SHOE", "SHRIMP",
    "SIGN", "SINK", "SKI", "SKUNK", "SKY", "SLEEVE", "SLIME", "SLUSH", "SMILE",
    "SMOKE", "SNAIL", "SNAKE", "SNOW", "SOAP", "SOCK", "SOUP", "SPARK", "SPEAR",
    "SPONGE", "SPOON", "SPRING", "SQUARE", "STAIR", "STAR", "STEAK", "STEAM",
    "STEM", "STICK", "STONE", "STOOL", "STORE", "STORM", "STOVE", "STRAW",
    "STREET", "STRING", "SUIT", "SUN", "SWAMP", "SWORD", "TAIL", "TANK",
    "TAPE", "TEA", "TEETH", "TENT", "THREAD", "THUMB", "TIE", "TOAD", "TOAST",
    "TOE", "TOOL", "TOOTH", "TOY", "TRAIN", "TRASH", "TRAY", "TREE", "TRUCK",
    "VAN", "VASE", "VEST", "VINE", "WALL", "WAND", "WAVE", "WEB", "WEED",
    "WHALE", "WHEEL", "WING", "WOLF", "WOOD", "WORLD", "WORM", "YARD", "ZOO"
]

# run this only once
#nltk.download("wordnet")
# Extract common English nouns from WordNet
#common_nouns = set(word.lower() for word in wn.all_lemma_names(pos='n'))


# Create 50 different lists of 30 nouns each
n_lists = 50  # Number of lists
list_length = 30
word_lists = [random.sample(list(common_nouns), list_length) for _ in range(n_lists)]


# run this only the first time
# model = api.load("word2vec-google-news-300")
# model.save("word2vec_model.bin")

# Otherwise, Load the saved model
model = KeyedVectors.load("word2vec_model.bin")


# from utils import softmax
def softmax(x, beta):
    """Compute the softmax function.
    :param x: Data
    :param beta: Inverse temperature parameter.
    :return:
    """
    x = np.array(x)
    return np.exp(beta * x) / sum(np.exp(beta * x))


def cosine_distance_matrix(words, beta, model):
    n = len(words)
    matrix = np.ones((n, n))  # Initialize identity matrix

    for i in range(n):
        for j in range(i + 1, n):  # Compute only upper triangle
            if words[i] in model and words[j] in model:
                similarity = model.similarity(words[i], words[j])
                matrix[i, j] = matrix[j, i] = similarity # Symmetric matrix

    cos_dist_probability = softmax(matrix, beta)

    return cos_dist_probability


# Define parameters. Potentially define parameters so that they fit typical behaviour  (average for participants)
# TODO: remove unused variables
alphas = [0.1, 0.5, 0.9]  # np.linspace(0, 1, 11)
n_trials = 1
alpha_thresh = 0.0001
c = -1
recall_threshold = 1
noise_std = 0.0003  # (0.0 - 0.8)
learning_rate = .1  # .1  # learning rate for SR
beta = 40
decay_time = 0  # 1000  # number of time steps delay between learning and recall


# Compute 50 matrices (one per word list)
cos_dist_as_probability = [cosine_distance_matrix(words, beta, model) for words in word_lists]


class Agent(object):
    """Simple SR learning agent with eligibility traces.
    """

    def __init__(self, ls, n_states, discount=0.95, decay=.5, learning_rate=.05, theta=0.5):  # .05):
        # parameters
        self.n = n_states
        self.gamma = discount  # this parameter controls the discount of future states.
        self.rho = decay  # this parameter controls the decay of the eligibility trace.
        self.lr = learning_rate
        self.theta = theta  # this parameter controls the strength of new inputs in the trace
        self.rho_encoding = 0.5
        self.theta_encoding = 0.5  # TODO: maybe encoding and retrieval parameters should be the same. Check.

        # initialise
        self.Mcs = np.zeros((self.n, self.n))  # the SR matrix
        self.Msc = np.eye(self.n)
        self.S = cos_dist_as_probability[ls]  # the SR matrix with semantic distances
        self.Q = np.zeros((self.n, self.n))  # weighted SR matrix
        self.trace = np.zeros(self.n)  # vector of eligibility traces
        self.SR_Ms = np.zeros((self.n, self.n, n_trials))
        self.all_retr = np.zeros((n_trials, self.n))
        self.x = np.zeros(self.n)  # FIXME: why is this here? what is it?
        self.rec = np.zeros((n_trials, self.n))
        self.Cin = None

    def update(self, state, next_state):
        self.trace = self.update_trace(state)
        self.Mcs = self.update_sr(next_state, state)

    def update_sr(self, next_state, state):
        error = np.eye(self.n)[next_state] + self.gamma * self.Mcs[next_state] - self.Mcs[state]
        return self.Mcs + self.lr * np.outer(self.trace, error)  # takes two vectors to produce a NxM matrix

    def compute_cin(self, state, backward_sampling=False):
        if backward_sampling:
            self.Msc = self.Mcs.transpose()
        else:
            self.Msc = np.eye(self.n)
        x = np.eye(self.n)[state]
        return self.Msc.dot(
            x)

    def update_trace(self, state, backward_sampling=False, recall_phase=False):
        if recall_phase:
            # Vector addiction of trace vector and state vector 0,0,...1...0,0
            self.Cin = self.compute_cin(state, backward_sampling=backward_sampling)
            return self.rho * self.trace + self.theta * self.Cin
        else:  # during encoding
            self.Cin = self.compute_cin(state)
            return self.rho_encoding * self.trace + self.theta_encoding * self.Cin
            # rho = deacay theta = strength of new inputs in the trace

    def decay_trace(self):
        self.trace = self.gamma * self.rho * self.trace

    def reset_trace(self):
        self.trace = np.zeros(self.n)

    def reset_matrix(self):
        self.Mcs = np.zeros((self.n, self.n))
        # self.Msc = np.eye(self.n)

    def reset_x(self):
        self.x = np.zeros(self.n)

    def reset(self):
        self.reset_trace()
        self.reset_matrix()
        self.reset_x()

    def do_free_recall_sampling(self, p_stop=.05, aph=1., backward_sampling=False):
        # Compute Q matrix as a weighted sum of Mcs and S
        self.Q = aph * self.Mcs + (1 - aph) * self.S
        already_recalled = np.zeros(self.n, dtype=bool)  # added from v3

        # Initialise trace and recall list
        recall_list = [np.nan] * self.n

        for i in range(self.n):
            if np.random.rand() < p_stop:
                break
            # Compute activation strengths
            f_strength = np.matmul(self.trace, self.Q)
            f_strength[already_recalled] = 0.001 #make it very small rather than 0

            if f_strength.sum() == 0:
                break
            # recall_prob = np.ones(len(f_strength))
            # recall_prob /= recall_prob.sum()
            else:
                recall_prob = f_strength / f_strength.sum()


            # Randomly select an item to recall
            recalled_word = np.random.choice(np.arange(len(recall_prob)), p=recall_prob)

            # Update trace and recall list
            self.trace = self.update_trace(recalled_word, backward_sampling=backward_sampling, recall_phase=True)
            already_recalled[recalled_word] = True
            recall_list[i] = recalled_word

        return recall_list

    def do_free_recall_probability(self, recalled_words, p_stop=.05, aph=1., backward_sampling=False):
        # Compute Q matrix as a weighted sum of Mcs and S
        self.Q = aph * self.Mcs + (1 - aph) * self.S
        already_recalled = np.zeros(self.n, dtype=bool)  # added from v3

        # Initialise trace and recall list
        recall_prob_list = np.full((1, self.n), np.nan)  # [] #[[np.nan] * self.n] * self.n

        for i in range(self.n):
            if np.random.rand() < p_stop:
                break
            # Compute activation strengths
            f_strength = np.matmul(self.trace, self.Q)
            f_strength[already_recalled] = 0.001 #make it very small rather than 0

            if f_strength.sum() == 0:
                break
            # recall_prob = np.ones(len(f_strength))
            # recall_prob /= recall_prob.sum()
            else:
                recall_prob = f_strength / f_strength.sum()

            #if np.any(recall_prob<0):
                #print("check this")
               # exit


            recalled_word = recalled_words[i]
            if not np.isnan(recalled_word):
                recall_prob_list[0, i] = recall_prob[recalled_word]
                # Update trace and recall list
                self.trace = self.update_trace(recalled_word, backward_sampling=backward_sampling, recall_phase=True)
                already_recalled[recalled_word] = True
            else:
                break

        return recall_prob_list


def generate_data(alpha, ls):
    """Generates recalled words and their probabilities for a given list."""
    word_list = word_lists[ls]
    ag = Agent(ls, len(word_list), learning_rate=learning_rate, decay=.95)
    word_index = {word: i for i, word in enumerate(word_list)}
    state_sequence = [word_index[word] for word in word_list]

    for trial in range(n_trials):
        ag.reset_trace()
        ag.reset_matrix()
        ag.reset_x()

        for t in range(len(state_sequence) - 1):
            ag.update(state_sequence[t], state_sequence[t + 1])

        ag.SR_Ms[:, :, trial] = ag.Mcs

        for t in range(decay_time):
            ag.decay_trace()

        recalled_words = ag.do_free_recall_sampling(p_stop=.05, aph=alpha)

        # Ensure probabilities are a NumPy array
       # recall_prob_words = np.array(recall_prob_words)

        # Pad recalled words to maintain consistent length
        while len(recalled_words) < ag.n:
            recalled_words.append(np.nan)

    return recalled_words


def llik_td_list(x, *args): #(ag, alpha, recalled_words):
    """Computes the negative log-likelihood for a recalled list."""

    alpha = x
    ag, recalled_words = args
    logp_recall = np.full(list_length, np.nan)  # Initialize with NaN
    word_list = word_lists[ls]
    words_recalled = np.array(recalled_words)

    word_index = {word: i for i, word in enumerate(word_list)}
    state_sequence = [word_index[word] for word in word_list]

    for trial in range(n_trials):
        ag.reset_trace()
        ag.reset_matrix()
        ag.reset_x()

        for t in range(len(state_sequence) - 1):
            ag.update(state_sequence[t], state_sequence[t + 1])

        ag.SR_Ms[:, :, trial] = ag.Mcs

        for t in range(decay_time):
            ag.decay_trace()


        words_prob = ag.do_free_recall_probability(recalled_words, p_stop=.05, aph=alpha)
        words_prob = np.squeeze(words_prob)
    #prob_words = np.squeeze(recall_prob)

    #if np.any(words_prob<0):
     #   print(f"check prob p={words_prob}")
      #  exit()

    for w in range(words_prob.shape[0]):
        try:
            if not np.all(np.isnan(words_prob[w])):  # ✅ Check for NaN before conversion
                #pw = int(words_recalled[w])
                if words_prob[w]<0:
                    print(f"Issue with prob p={words_prob[w]} for w ={w}")
                logp_recall[w] = np.log(words_prob[w] + 1e-10)  # Avoid log(0)
        except IndexError as e:
            print(f"IndexError at w={w} : {e}")

    return -np.nansum(logp_recall)


def llik_td(x, *args):
    """Computes the total negative log-likelihood across all lists."""
    alpha = x
    all_recalled_words = args
    logp_ns_lists = np.zeros(n_lists)
    all_recalled_words = np.array(all_recalled_words, dtype=object)
    all_recalled_words = np.squeeze(all_recalled_words)

    for ls in range(n_lists -1):
        ag = Agent(ls, list_length, learning_rate=learning_rate, decay=.95)
        logp_ns_lists[ls] = llik_td_list(alpha, ag, all_recalled_words[ls])

    return np.nansum(logp_ns_lists)


if __name__ == '__main__':
    """Main execution loop that avoids duplicate calls to generate_data."""


    # generate some data

    all_recalled_trials = []
    for a in range(len(alphas)):

        alpha = alphas[a]


        logp_ns_all = np.zeros(n_lists)
        all_recalled_words = []

        for ls in range(n_lists - 1):
            # initialise agent:
            ag = Agent(ls, list_length, learning_rate=learning_rate, decay=.95)

            recalled_words = generate_data(alpha, ls)
            logp_ns_all[ls] = llik_td_list(alpha, ag, recalled_words)


            all_recalled_words.append(recalled_words)
            #all_recalled_probs.append(recall_prob)

        # Convert lists to NumPy arrays for consistency
        all_recalled_words = np.array(all_recalled_words, dtype=object)
        # all_recalled_probs = np.array(all_recalled_probs, dtype=object)
        all_recalled_trials.append(all_recalled_words)

    inferred_alpha = np.zeros(len(alphas))
    logp_ns = np.zeros(len(alphas))

    for a in range(len(alphas)):

        alpha = alphas[a]
        all_recalled_words = all_recalled_trials[a]

        # Compute log-likelihood only ONCE using precomputed recall data
        logp_ns[a] = llik_td(alpha, all_recalled_words)

        result = scipy.optimize.minimize(llik_td, alpha, args=(all_recalled_words), bounds=[(0, 1)])
        inferred_alpha[a] = result.x[0]

        print(result)
        print("")
        print(f"MLE: alpha = {result.x[0]:.2f} (true value = {alpha})")
        print("end")

    print('done')

    g = plt.figure(1)
    x = np.array(alphas)
    y = np.array(inferred_alpha)
    plt.xlabel("True alpha")
    plt.ylabel("Estimated alpha")
    # plt.xticks(np.arange(0, 1, 0.1))
    # plt.yticks(np.arange(0, 1, 0.1))
    plt.plot(x, y, 'o')
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b)
    g.show()

    print("wtf")




