import numpy as np
import matplotlib.pyplot as plt


n_items = 2
tau = .5
kappa = 1.2
lambda_param = .9
threshold = 1.
noise_std = .2

n_time_steps = 30

L = np.ones((n_items, n_items)) - np.eye(n_items)


all_x = np.empty((n_time_steps, n_items))

x = np.zeros(n_items)
for t in range(n_time_steps):

    f = np.eye(n_items)[np.random.choice(range(n_items))]  # these are the inputs! now a random [1,0] vector every time

    coef_var = np.std(f) / f.mean()

    noise = np.random.normal(0, noise_std, n_items)
    x_new = (1 - tau * kappa - tau * lambda_param * L) @ x + coef_var * tau * f + noise
    x = np.maximum(x_new, 0)

    all_x[t, :] = x


plt.plot(all_x)
plt.axhline(threshold, color='k', linestyle='--')
plt.show()