import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from psifr import fr
from itertools import product

from SR_MSweighted_v3 import Agent


# Define parameters

params = {
    'alphas': [1, 0],
    'alpha_thresh': 0.0001,
    'c': -1,
    'recall_threshold': 1,
    'noise_std': 0.0003,
    'learning_rate': 0.1,  # learning rate for SR
    'decay_const': .5,  # eligibility trace decay parameter
    'discount': .5,
    'beta': .5,
    'decay_time': 1000  # number of time steps delay between learning and recall
}

n_sims = 500


# some example input words:
text_body = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 1 2 3 4'
word_list = text_body.split()
sequence_length = len(word_list)
# associate each word with a unique state index:

word_list = pd.DataFrame({
    'item': word_list,
    'token': np.arange(len(word_list)),  # tokenization for the model (i.e. which state index corresponds to which word)
    'input': np.arange(len(word_list)) + 1
})


def write_simulation_output_to_df_for_psifr(all_recalled_words, word_list):
    # construct dataframe with all model results in the format required for psifr
    results_df = pd.concat([word_list] * n_sims)
    results_df['subject'] = np.repeat(np.arange(n_sims), len(word_list))
    results_df['recall'] = np.concatenate([np.isin(word_list.token, recall) for recall in all_recalled_words])  # recalled or not
    results_df['output'] = np.concatenate([[recall.index(tok) if tok in recall else np.nan for tok in word_list.token] for recall in all_recalled_words]) # position in recall for each word
    results_df['list'] = 1
    results_df['study'] = True
    return results_df


def simulate_learning_and_recall(word_list, n_sims, params):
    ag = Agent(len(word_list), learning_rate=params['learning_rate'], decay=params['decay_const'], discount=params['discount'], beta=params['beta'])
    p_stop = .05
    all_recalled_words = []

    for trial in range(n_sims):
        ag.reset()
        # learning over M does not accumulate across different trials (each trial is tested with free recall
        # individually) (this is because we're modelling each of these as a different run of the model / different subject).

        for n_readings in range(10):
            for t in word_list.token[:-1]:
                ag.update(t, t + 1)

        ag.SR_Ms[:, :, trial] = ag.Mcs

        for t in range(params['decay_time']):
            ag.decay_trace()

        recalled_words = ag.do_free_recall_sampling(p_stop, backward_sampling=False)
        all_recalled_words.append(recalled_words)
    return write_simulation_output_to_df_for_psifr(all_recalled_words, word_list)


params['decay_const'] = 1.
params['beta'] = 0.
discounts = [0., .5]

crp_ls = []
for gamma in discounts:
    params['discount'] = gamma
    sim_result = simulate_learning_and_recall(word_list, n_sims, params)
    crp = fr.lag_crp(sim_result)
    crp['discount'] = gamma
    crp_ls.append(crp)


a = fr.plot_lag_crp(pd.concat(crp_ls), col='discount')
