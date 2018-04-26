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
    #power = 0.0015
    power = 0.001

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



@numba.jit(nopython=True)
def step_cartpole(prev_state, action):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5 # actually half the pole's length
    polemass_length = (masspole * length)
    force_mag = 10.0
    tau = 0.02  # seconds between state updates

    # Angle at which to fail the episode
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4


    force = min(max(action[0], -force_mag), force_mag)

    x, x_dot, theta, theta_dot = prev_state


    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
    xacc  = temp - polemass_length * thetaacc * costheta / total_mass
    x  = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc

    state = (x,x_dot,theta,theta_dot)
    done =  x < -x_threshold \
            or x > x_threshold \
            or theta < -theta_threshold_radians \
            or theta > theta_threshold_radians
    done = bool(done)

    if not done:
        reward = 1.0
    else:
        reward = 0.0

    return np.array(state), reward, done, None
