import numpy as np
import matplotlib.pyplot as plt
from itertools import product

n_items = 10
tau = .5
kappa = .5  # self inhibition
lambda_param = 8.  # lateral inhibition
threshold = 1.
noise_std = 0

n_time_steps = 50

L = np.ones((n_items, n_items)) - np.eye(n_items)


def accumulate_one_step(old_x, input_strength, params):
    coef_var = np.std(input_strength) / input_strength.mean()
    noise = np.random.normal(0, params['noise_std'], params['n_items'])
    x_new = (1 - params['tau'] * params['kappa'] - params['tau'] * params['lambda'] * L).T @ old_x \
            + coef_var * params['tau'] * input_strength + noise
    return np.maximum(x_new, 0)


def run_accumulator(n_steps, parameters, distractor_activation=.2):
    # initialize x
    x = np.zeros(parameters['n_items'])
    x_in_time = np.empty((n_steps + 1, n_items))
    x_in_time[0] = x
    for t in range(1, n_steps + 1):
        f_strength = np.eye(parameters['n_items'])[0] + (np.ones(parameters['n_items']) -
                                                         np.eye(parameters['n_items'])[0]) * distractor_activation  # note these inputs!
        x = accumulate_one_step(x, f_strength, parameters)
        x_in_time[t, :] = x
    return x_in_time


if __name__ == '__main__':

    params = {
        'n_items': n_items,
        'kappa': kappa,
        'lambda': lambda_param,
        'tau': tau,
        'threshold': threshold,
        'noise_std': noise_std
    }

    n_rows = 5
    n_cols = 5

    k_min, k_max = (.1, 1.4)
    l_min, l_max = (.1, 1.8)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 9), sharex=True, sharey=True)

    kappa_range = np.linspace(k_min, k_max, n_rows)
    lambda_range = np.linspace(l_min, l_max, n_cols)

    i = 0
    for k, l in product(kappa_range, lambda_range):
        params['kappa'] = k
        params['lambda'] = l
        all_x = run_accumulator(n_time_steps, params)
        row, col = np.unravel_index(i, (n_rows, n_cols))
        axes[row, col].plot(all_x)
        axes[row, col].set_ylim([0, 1.2])
        plt.sca(axes[row, col])
        plt.axhline(params['threshold'], color='k', linestyle='--')

        axes[row, col].set_ylabel('kappa: {:.2f}'.format(k))
        axes[row, col].set_xlabel('lambda: {:.2f}'.format(l))

        i += 1

    plt.tight_layout()
    plt.savefig('accumulator_with_different_params.png')
    plt.show()

    # Now check for a requirement that one but only one accumulator reaches threshold
    k_min, k_max = (.1, 3.)
    l_min, l_max = (.1, 3.)

    n_rows = 40
    n_cols = 40
    n_z = 11

    kappa_range = np.linspace(k_min, k_max, n_rows)
    lambda_range = np.linspace(l_min, l_max, n_cols)
    distractor_range = np.linspace(0., 1., n_z)

    param_grid = np.zeros((n_rows, n_cols, n_z), dtype=bool)
    i = 0
    for k, l, dist in product(kappa_range, lambda_range, distractor_range):
        params['kappa'] = k
        params['lambda'] = l

        all_x = run_accumulator(n_time_steps, params, distractor_activation=dist)
        idx = np.unravel_index(i, (n_rows, n_cols, n_z))

        max_accum_value = all_x.max(axis=0)  # get max for each item
        good_param = sum(max_accum_value >= params['threshold']) == 1
        param_grid[idx] = good_param
        i += 1

    # the code below plots a matrix of true and false values where true means only one item crossed
    # the threshold. This means that you can see the parameter values for which this is true.
    # i also plotted that for 10 different values for the "distractor" strength.
    xx, yy = np.meshgrid(kappa_range, lambda_range)
    for i, d in enumerate(distractor_range):
        plt.figure()
        plt.pcolormesh(xx, yy, param_grid[:, :, i])
        plt.xlabel('Kappa')
        plt.ylabel('Lambda')
        plt.savefig('good_param_vals{:.2f}.png'.format(d))
        plt.show()
