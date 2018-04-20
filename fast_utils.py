import numpy as np
import math
import numba
from utils import apply_along_axis2

@numba.jit(nopython=True)
def step_mountain_car(prev_state, action):
    min_action = -1.0
    max_action = 1.0
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    power = 0.0015

    position = prev_state[0]
    velocity = prev_state[1]
    force = min(max(action[0], -1.0), 1.0)

    velocity += force*power -0.0025 * math.cos(3*position)
    if (velocity > max_speed): velocity = max_speed
    if (velocity < -max_speed): velocity = -max_speed
    position += velocity
    if (position > max_position): position = max_position
    if (position < min_position): position = min_position
    if (position==min_position and velocity<0): velocity = 0

    done = bool(position >= goal_position)

    reward = 0
    if done:
        reward = 100.0
    reward-= math.pow(action[0],2)*0.1

    new_state = np.zeros(2)
    new_state[0] = position
    new_state[1] = velocity
    # new_state = np.array([position, velocity])
    return new_state, reward, done, None

@numba.jit(nopython=True)
def fast_calc_score_theta(actions, features, theta_mat, inv_cov):
    batch_size = actions.shape[0]
    H = actions.shape[1]
    feat_dim = features.shape[2]

    scores = np.zeros((batch_size, H, feat_dim))

    for b in range(batch_size):
        for h in range(H):
            phi = features[b,h].T
            a = actions[b,h][0]

            mu = np.dot(theta_mat,phi)
            scores[b,h] = inv_cov * ((a - mu) * (phi.T))

    return scores


@numba.jit(nopython=True)
def fast_calc_score_sigma(actions, features, theta_mat, cov):
    batch_size = actions.shape[0]
    H = actions.shape[1]

    scores = np.zeros((batch_size, H))

    for b in range(batch_size):
        for h in range(H):
            phi = features[b,h].T
            a = actions[b,h][0]

            mu = np.dot(theta_mat,phi)[0]

            scores[b,h] = ((a-mu)**2 - cov)/(cov * np.sqrt(cov))


    return scores
