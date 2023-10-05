import numpy as np
import gymnasium as gym
import gymnasium as gym
from sklearn.preprocessing import KBinsDiscretizer

lower_bound, upper_bound = [-.5, -3, -.2, -4], [.5, 3, .2, 4]
scaler = np.array([100, 10, 100, 10])
shift = np.array([15,10,1,10])
norm = np.array([0.25, 0.25, 0.01, 0.1])

def fixed_discretized_state(state, n_bins=None):
    state_bins = [
        [-1.5, 1.5], # position bins
        [-.5, .5], # speed bins
        [-.1, -.05, -.03, -.01,  0, .01, 0.03, .05, .1], # angle bins
        [-1, -.75, -.5, -.25, 0, .25, .5, .75, 1]  # angle velocity bins
    ]
    return tuple([np.digitize(s, bins) for s, bins in zip(state, state_bins)])

def simple_discretized_state(state, n_bins=(10, 10, 10, 10)):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretizer.fit([lower_bound, upper_bound])
    discrete_state = discretizer.transform([state])[0]
    return tuple(map(int, discrete_state))

def discretize_state_scaler(state, n_bins=None):
    scaled_state = state*scaler
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretizer.fit([lower_bound, upper_bound])
    discrete_state = discretizer.transform([scaled_state])[0]
    return tuple(discrete_state.astype(int))

def discretize_state_normalized(state, n_bins=None):
    discrete_state = state/norm + shift
    return tuple(discrete_state.astype(int))

def discretize_state_min_max(observation, n_bins=(10, 10, 10, 10)):
    min_max = (np.array(observation) - np.array(lower_bound)) / (np.array(upper_bound) - np.array(lower_bound))
    discretized_state = min_max * np.array(n_bins)
    intervals = [np.linspace(low, high, num) for (low, high), num in zip(zip(lower_bound, upper_bound), np.array(n_bins)-1)]
    return tuple([np.digitize(i, bins) for i, bins in zip(discretized_state.astype(int), intervals)])

def discretize_state_croped(state, n_bins=(6, 12)):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretizer.fit([lower_bound[2:], upper_bound[2:]])
    discrete_state = discretizer.transform([[state[2], state[3]]])[0]
    return tuple(map(int, discrete_state))



