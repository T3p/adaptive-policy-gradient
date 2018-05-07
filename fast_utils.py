import numpy as np
import math
import numba
from utils import apply_along_axis2

"""classic Acrobot task"""
from gym import core, spaces
from gym.utils import seeding
from numpy import sin, cos, pi

def normalize_cartpole_rllab(s):
    return np.array([s[0] / 0.1, s[1] / 0.5, s[2] / 0.5, s[3]])

def normalize_cartpole(s):
    a = 12 * 2 * math.pi / 360
    return np.array([s[0] / 4.8, s[1] / 5, s[2] / a, s[3] / 5])

def normalize_pendulum(s):
    return np.array([s[0], s[1], s[2]/8.0])

def make_phi_pendulum():
    means = np.array([[-1, -1, -8 ],
               [-1 , -1, 0],
               [-1 , -1,  8],
               [-1 , 0, -8 ],
               [-1 , 0, 0 ],
               [-1 , 0, 8 ],
               [-1 , 1, -8 ],
               [-1 , 1, 0 ],
               [-1 , 1, 8 ],
               [0, -1, -8],
               [0, -1, 0],
               [0, -1, 8],
               [0, 0, -8],
               [0, 0, 0],
               [0, 0, 8],
               [0, 1, -8],
               [0, 1, 0],
               [0, 1, 8],
               [1, -1, -8],
               [1, -1, 0],
               [1, -1, 8],
               [1, 0, -8],
               [1, 0, 0],
               [1, 0, 8],
               [1, 1, -8],
               [1, 1, 0],
               [1, 1, 8]])

    @numba.jit(nopython=True)
    def f(s):
        phi = np.zeros(means.shape[0])
        for i in range(means.shape[0]):
            phi[i] = gauss_kernel(s, means[i])

        return phi

    return f

def make_phi_mountain_car():
    means = np.array([[-1.2 , -0.07 ],
               [-1.2 , 0 ],
               [-1.2 ,  0.07 ],
               [-0.6 ,  -0.07 ],
               [-0.6, -0 ],
               [-0.6, 0.07 ],
               [0,  -0.07 ],
               [0,  0 ],
               [ 0 , 0.07 ],
               [ 0.6 , -0.07 ],
               [ 0.6 ,  0 ],
               [ 0.6 ,  0.07 ]])

    @numba.jit(nopython=True)
    def f(s):
        phi = np.zeros(means.shape[0])
        for i in range(means.shape[0]):
            phi[i] = gauss_kernel(s, means[i])

        return phi

    return f


def normalize_mountain_car(s):
    return np.array([(s[0] + 0.3) / 0.9,s[1] / 0.07])


@numba.jit(nopython=True)
def gauss_kernel(s1, s2):
    return math.exp(-np.linalg.norm(np.subtract(s1,s2), 2)**2)


@numba.jit(nopython=True)
def step_mountain_car(prev_state, action):
    min_action = -1.0
    max_action = 1.0
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    power = 0.0015
    # power = 0.001

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
    reward-= 0.001#math.pow(action[0],2)*0.1

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



@numba.jit(nopython=True)
def step_cartpole_rllab(prev_state, action):
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
        reward = 10.0 - (1 - math.cos(theta_dot)) - 10e-5 * (action[0])**2
    else:
        reward = 0.0

    return np.array(state), reward, done, None






@numba.jit(nopython=True)
def step_acrobot(prev_obs, action):
    dt = .2

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi


    torque_noise_max = 0.


    prev_state = np.zeros(4)
    prev_state[0] = np.arctan2(prev_obs[1], prev_obs[0])
    prev_state[1] = np.arctan2(prev_obs[3], prev_obs[2])
    prev_state[2] = prev_obs[4]
    prev_state[3] = prev_obs[5]

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    s = prev_state
    #torque = self.AVAIL_TORQUE[a]
    torque = min(max(action[0], -1), 1)


    # Now, augment the state with our force action so it can be passed to
    # _dsdt
    # s_augmented = np.append(s, torque)
    # print(s_augmented)
    s_augmented = np.zeros(s.shape[0] + 1)
    for i in range(s.shape[0]):
        s_augmented[i] = s[i]
    s_augmented[s.shape[0]] = torque

    #ns = rk4(_dsdt, s_augmented, [0, dt])
    y0 = s_augmented
    t = [0,dt]

    Ny = len(y0)
    yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0


    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        # x = derivs(y0, thist)
        # print(x)


        k1 = np.array(_dsdt(y0, thist))
        k2 = np.array(_dsdt(y0 + dt2 * k1, thist + dt2))
        k3 = np.array(_dsdt(y0 + dt2 * k2, thist + dt2))
        k4 = np.array(_dsdt(y0 + dt * k3, thist + dt))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    ns = yout

    # only care about final timestep of integration returned by integrator
    ns = ns[-1]
    ns = ns[:4]  # omit action
    # ODEINT IS TOO SLOW!
    # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
    # self.s_continuous = ns_continuous[-1] # We only care about the state
    # at the ''final timestep'', self.dt

    ns[0] = wrap(ns[0], -pi, pi)
    ns[1] = wrap(ns[1], -pi, pi)
    ns[2] = min(max(ns[2], -MAX_VEL_1), MAX_VEL_1)
    ns[3] = min(max(ns[3], -MAX_VEL_2), MAX_VEL_2)

    state = ns
    s = state

    terminal = bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)
    reward = -1. if not terminal else 0.

    state = np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])

    return (state, reward, terminal, None)



@numba.jit(nopython=True)
def _dsdt(s_augmented, t):
    book_or_nips = "book"
    m1 = 1.
    m2 = 1.
    l1 = 1.
    lc1 = 0.5
    lc2 = 0.5
    I1 = 1.
    I2 = 1.
    g = 9.8
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1 = s[0]
    theta2 = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]
    d1 = m1 * lc1 ** 2 + m2 * \
        (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
    phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
           - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
        + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2

    if book_or_nips == "nips":
        # the following line is consistent with the description in the
        # paper
        ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
            (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    else:
        # the following line is consistent with the java implementation and the
        # book
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
            / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)



@numba.jit(nopython=True)
def wrap(x, m, M):
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


@numba.jit(nopython=True)
def step_pendulum(prev_obs, a):
    max_speed=8
    max_torque=2.
    dt=.05

    th = np.arctan2(prev_obs[1], prev_obs[0])
    thdot = prev_obs[2]

    g = 10.
    m = 1.
    l = 1.

    # u = np.clip(u, -self.max_torque, self.max_torque)[0]

    u = min(max(a[0], -max_torque), max_torque)

    last_u = u # for rendering
    th_normalize = (((th+np.pi) % (2*np.pi)) - np.pi)

    costs = th_normalize**2 + .1*thdot**2 + .001*(u**2)

    newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
    newth = th + newthdot*dt
    newthdot = min(max(newthdot, -max_speed), max_speed) #pylint: disable=E1111


    new_obs = np.array([np.cos(newth), np.sin(newth), newthdot])
    return new_obs, -costs, False, None
