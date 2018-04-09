from numpy.random import normal
import numpy as np
import math
from scipy.linalg import sqrtm
import time

"""Policies"""


class GaussPolicy:
    """Gaussian parametric policy with mean linear in the features and constant variance matrix"""

    def __init__(self,theta,cov):
        """Parameters:
        theta -- parameter vector of size d*m, representing a d x m parameter matrix in lexographical order,
            where d is the dimensionality of the action space and m the size of the feature vector
        cov -- d X d covariance matrix (variance in the scalar case)
        """

        #cov = np.asmatrix(cov)
        cov = np.atleast_2d(cov)
        cov = np.power(cov, 2)

        assert np.array_equal(cov,np.transpose(cov))
        self.cov = cov.astype(float)
        d = self.act_dim = np.shape(cov)[0]
        self.sigma = math.sqrt(np.linalg.det(cov))

        self.inv_cov = np.linalg.inv(self.cov)

        theta = np.atleast_1d(theta)
        self.param_len = theta.size
        assert np.size(theta)%d==0
        self.feat_dim = m = np.size(theta)//d
        self.theta_mat = np.reshape(theta,(d,m)).astype(float)

    def get_theta(self):
        """Returns: policy parameter as a vector representing parameter matrix in lexographical order"""
        return np.ravel(self.theta_mat)

    def act(self,phi,noise=None,deterministic=False):
        """Policy stochastic mapping

        Parameters:
        phi -- feature vector of state s
        noise (optional) -- gaussian noise realization

        Returns: action a sampled from pi(a|s), or the expected value if deterministic==True
        """

        #Expected value
        mu = np.dot(self.theta_mat,phi)

        if deterministic:
            return mu

        #Gaussian noise
        if noise is None:
            noise = normal(0,1,self.act_dim)

        a = mu + np.dot(np.linalg.cholesky(self.cov),noise)
        return np.asscalar(a) if np.size(a)==1 else np.ravel(a)

    def prob(self,a,phi):
        """Policy density function

        Parameters:
        phi -- feature vector of state s
        a -- action to evaluate

        Returns: probability density pi(a|s)
        """

        phi = np.asmatrix(phi).T
        a = np.asmatrix(a).T

        mu = np.dot(self.theta_mat,phi)
        normalization = 1.0/math.sqrt((2*math.pi)**self.act_dim*np.linalg.det(self.cov))
        return normalization*math.exp(-0.5*np.dot((a-mu).T,np.dot(np.linalg.inv(self.cov),(a-mu))))

    def score(self,a,phi):
        """Score function

        Parameters:
        phi -- feature vector of  state s
        a -- action

        Returns: the gradient w.r.t. theta of the logarithm of the policy density function pi(a|s)
            as a flat array of length d*m, representing a d x m matrix in lexicographical order
        """

        # phi = np.asmatrix(phi).T
        # a = np.asmatrix(a).T
        phi = np.atleast_2d(phi).T
        a = np.atleast_2d(a).T

        # score = np.dot(np.linalg.inv(self.cov), \
        #             np.dot((a - np.dot(self.theta_mat,phi)),np.transpose(phi)))
        score = np.dot(self.inv_cov, \
                    np.dot((a - np.dot(self.theta_mat,phi)),np.transpose(phi)))

        return np.asscalar(score) if np.size(score)==1 else np.ravel(score)


    def penaltyCoeff(self,R,M,gamma,volume,c=None):
        """Penalty coefficient for performance improvement bounds

        Parameters:
        R -- maximum absolute-value reward
        M -- upper bound on all state features
        gamma -- discount factor
        volume -- volume of the action space
        """
        if not c==None:
            return c

        return float(R*M**2)/((1-gamma)**2*self.sigma**2)* \
            (float(volume)/math.sqrt(2*math.pi*self.sigma**2) + \
                float(gamma)/(2*(1-gamma)))

    def update(self,delta_theta):
        """Updates the policy parameter

        Parameters:
        delta_theta -- vector that is summed to the policy parameter
        """
        assert np.size(delta_theta)==self.param_len
        theta_new = self.get_theta() + np.atleast_1d(delta_theta)
        self.theta_mat = np.reshape(theta_new,(self.act_dim,self.feat_dim)).astype(float)


class ExpGaussPolicy(GaussPolicy):
    """Scalar implementation of a Gaussian Policy with variance parameterized with an exponential function
    """
    def __init__(self,theta,w):
        assert(np.isscalar(w))
        self.w = w
        super().__init__(theta, np.exp(w))

    def update_w(self, deltaW):
        assert(np.isscalar(deltaW))
        self.w += deltaW

        #self.cov = np.asmatrix(math.exp(self.w))
        self.cov = np.atleast_2d(math.exp(self.w))
        self.cov = np.power(self.cov, 2)

        self.sigma = math.sqrt(np.linalg.det(self.cov))
        self.inv_cov = np.linalg.inv(self.cov)

    def score_sigma(self, a, phi):
        """Score function w.r.t. sigma

        Parameters:
        a -- action
        phi -- feature vector of state s

        Returns: the gradient w.r.t. sigma of the logarithm of the Policy
        """
        # phi = np.asmatrix(phi).T
        # a = np.asmatrix(a).T

        phi = np.atleast_2d(phi).T
        a = np.atleast_2d(a).T

        return np.asscalar(((a-self.theta_mat*phi)**2 - self.cov)/(self.cov * np.sqrt(self.cov)))

    def penaltyCoeffSigma(self,R,M,gamma,volume,d=None):
        """Penalty coefficient for sigma for performance improvement bounds

        Parameters:
        R -- maximum absolute-value reward
        M -- upper bound on all state features
        gamma -- discount factor
        volume -- volume of the action space
        """
        if not d==None:
            return d

        sigma = self.sigma
        c = (4*(math.sqrt(7) - 2)*math.exp((math.sqrt(7))/(2) - 2)) / (math.sqrt(2*math.pi))
        return R/((1-gamma)**2) * ((c*volume) / (2*sigma) + (gamma) / ((1-gamma)))
