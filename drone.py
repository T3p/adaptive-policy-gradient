from lqg import LQG
import gym
from gym import spaces
import numpy as np


#Register if not already registered
from gym.envs.registration import register, spec
try:
    spec('Drone-v0')
except:
    register(id='Drone-v0',
            entry_point='drone:Drone')


class Drone(LQG):
    
    def __init__(self, discrete_reward=False):
        self.discrete_reward = discrete_reward
        self.max_pos = 40. #Target height [m]
        self.max_action = 15. #Max vertical thrust [N]
        drone_mass = self.mass = 10. #[Kg]
        load_mass = 0. #[Kg]
        self.sigma_noise = np.zeros((1,3)) #Thrust controller noise
        freq = 1. #Control frequency [Hz]
        self.horizon = 100 #Control steps
        
        self.gamma = 1 - 1./self.horizon
        tau = self.tau = 1./freq
        mass = drone_mass + load_mass
        pos_penalty = 1.
        vel_penalty = 100.
        force_penalty = 0.1
        self.A = np.array([[1., tau, 0.], 
                           [0.,  1., -9.8*tau/mass],
                           [0., 0., 1.]])
        self.B = np.array([[0., 0., 0.],
                          [tau/mass, 0., 0.],
                          [0., 0., 0.]])
        self.Q = np.array([[pos_penalty, 0., 0.],
                           [0., vel_penalty, 0.],
                           [0., 0., 0.]])
        self.R = force_penalty*np.eye(3)
        self.initial_states = []

        # gym attributes
        self.viewer = None
        high = np.array([self.max_pos])
        self.action_space = spaces.Box(low=-self.max_action,
                                            high=self.max_action,
                                            shape=(1,))
        self.observation_space = spaces.Box(low=-high,
                                            high=high)

        # initialize state
        self.seed()
        self.reset()

    def computeOptimalParam(self):
        return super(Drone, self).computeOptimalK()[0]

    def step(self, action, render=False):
        u = np.concatenate((np.atleast_1d(action), [0., 0.]))
        return super(Drone, self).step(u, render)

    def reset(self, state=None):
        x = -self.max_pos 
        #x = super(Drone, self).reset(state)
        self.state = np.concatenate((np.atleast_1d(x), [0., 1.]))
        return self.state
    
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_height = (self.max_pos * 2) * 2
        scale = screen_height / world_height
        ballx = screen_width/2
        ballradius = 3

        if self.viewer is None:
            clearance = 0  # x-offset
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            mass = rendering.make_circle(ballradius * 2)
            mass.set_color(.8, .3, .3)
            mass.add_attr(rendering.Transform(translation=(clearance, 0)))
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)
            self.track = rendering.Line((ballx,0), (ballx, screen_height))
            self.track.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(self.track)
            self.ground = rendering.Line((0,screen_height/4), (screen_width, screen_height/4))
            self.ground.set_color(0.3, 0.8, 0.3)
            self.viewer.add_geom(self.ground)
            zero_line = rendering.Line((0, screen_height / 2),
                                       (screen_width, screen_height / 2))
            zero_line.set_color(0.3, 0.3, 0.8)
            self.viewer.add_geom(zero_line)

        y = self.state[0]
        bally = y * scale + screen_height / 2.0
        self.masstrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

if __name__ == '__main__':
    env = Drone()
    theta_star = env.computeOptimalK()
    print('theta^* = ', theta_star)
    print('J^* = ', env.computeJ(theta_star,env.sigma_noise))
