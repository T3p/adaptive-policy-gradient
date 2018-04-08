import numpy as np
from utils import *

"""Policy gradient estimation algorithms"""

class Estimator:

    def __init__(self,estimator_name='gpomdp'):
        """Parameters:
        estimator_name: algorithm used to estimate the gradient
        """
        estimators = {'reinforce':self.reinforce, 'gpomdp':self.gpomdp, 'gpomdp_w':self.gpomdp_w}
        self.estimate = estimators[estimator_name]
        self.estimate_w = estimators[estimator_name + '_w']

    def reinforce(self,features,actions,rewards,gamma,pol,use_baseline=True,average=True):
        """Batch REINFORCE policy gradient estimator with variance-minimizing baseline

        Parameters:
        features -- N x H x m array containing feature vectors for N episodes of length H
        actions -- N x H x d array containing actions for N episodes of length H
        rewards -- N x H array containing rewards for N episodes of length H
        gamma -- discount factor
        use_baseline -- if False, a baseline of b=0 is used instead

        Returns:
        the averaged gradient estimate if average==True, an array containing the N estimates otherwise
        """

        #Data
        assert features.shape[:2]==actions.shape[:2]==rewards.shape[:2]
        N = features.shape[0]
        H = features.shape[1]
        m = features.shape[2] if len(features.shape)>2 else 1

        #Q function
        disc_rewards = discount(rewards,gamma)
        q = np.sum(disc_rewards,1)

        #Eligibility vector
        scores = apply_along_axis2(pol.score,2,actions,features)
        sum_of_scores = np.sum(scores,1)

        #Optimal baseline

        b = np.zeros(m)
        if use_baseline and N>1:
            den = np.asarray(np.mean(sum_of_scores**2,0))
            np.putmask(den,den==0,1)
            b = np.mean(((sum_of_scores**2).T*q).T,0)/den

        #Gradient
        estimates = (sum_of_scores.T*q).T - sum_of_scores*b

        return np.mean(estimates,0) if average else estimates


    def gpomdp(self,features,actions,rewards,gamma,pol,use_baseline=True,average=True):
        """Batch G(PO)MDP policy gradient estimator with variance-minimizing baseline

        Parameters:
        features -- N x H x m array containing feature vectors for N episodes of length H
        actions -- N x H x d array containing actions for N episodes of length H
        rewards -- N x H array containing rewards for N episodes of length H
        gamma -- discount factor
        use_baseline -- if False, a baseline of b=0 is used instead

        Returns:
        the averaged gradient estimate if average==True, an array containing the N estimates otherwise
        """
        #Data
        assert features.shape[:2]==actions.shape[:2]==rewards.shape[:2]
        N = features.shape[0]
        H = features.shape[1]
        m = features.shape[2] if len(features.shape)>2 else 1

        #Q function
        disc_rewards = discount(rewards,gamma)

        #Eligibility vector
        scores = apply_along_axis2(pol.score,2,actions,features)
        cum_scores = np.cumsum(scores,1)

        #Optimal baseline:
        b = np.zeros((H,m))
        if use_baseline and N>1:
            den = np.mean(cum_scores**2,0)
            np.putmask(den,den==0,1)
            b = np.mean(((cum_scores**2).T*disc_rewards.T).T,0)/den

        #gradient estimate:
        estimates =  np.sum((cum_scores.T*disc_rewards.T).T - cum_scores*b,1)
        return np.mean(estimates,0) if average else estimates

    def gpomdp_w(self,features,actions,rewards,gamma,pol,use_baseline=True,average=True):
        """Batch G(PO)MDP policy gradient estimator with variance-minimizing baseline

        Parameters:
        features -- N x H x m array containing feature vectors for N episodes of length H
        actions -- N x H x d array containing actions for N episodes of length H
        rewards -- N x H array containing rewards for N episodes of length H
        gamma -- discount factor
        use_baseline -- if False, a baseline of b=0 is used instead

        Returns:
        the averaged gradient estimate if average==True, an array containing the N estimates otherwise
        """
        #Data
        assert features.shape[:2]==actions.shape[:2]==rewards.shape[:2]
        N = features.shape[0]
        H = features.shape[1]
        m = features.shape[2] if len(features.shape)>2 else 1

        #Q function
        disc_rewards = discount(rewards,gamma)

        #Eligibility vector
        scores = apply_along_axis2(pol.score_w,2,actions,features)
        cum_scores = np.cumsum(scores,1)

        #Optimal baseline:
        b = np.zeros((H,m))
        if use_baseline and N>1:
            den = np.mean(cum_scores**2,0)
            np.putmask(den,den==0,1)
            b = np.mean(((cum_scores**2).T*disc_rewards.T).T,0)/den

        #gradient estimate:
        estimates =  np.sum((cum_scores.T*disc_rewards.T).T - cum_scores*b,1)
        return np.mean(estimates,0) if average else estimates

class HEstimator(object):
    """Estimates h(θ,σ) = E~pθ[∇σlogpθ ∇θlogpθ R(τ)]
    """
    def __init__(self, name = 'naive'):
        estimators = {'naive': self.naive, 'gpomdp': self.gpomdp, 'gpomdp2': self.gpomdp2}
        self.estimate = estimators[name]

    def naive(self, features, actions, rewards, gamma, pol, average=True):
        scores_theta = apply_along_axis2(pol.score,2,actions,features)
        scores_w = apply_along_axis2(pol.score_w,2,actions,features)

        sum_1 = np.sum(scores_theta, 1)
        sum_2 = np.sum(scores_w, 1)

        disc_rewards = discount(rewards,gamma)
        sum_3 = np.sum(disc_rewards, 1)


        estimates = sum_1 * sum_2 * sum_3
        return np.mean(estimates, 0) if average else estimates

    def gpomdp(self, features, actions, rewards, gamma, pol, average=True):
        scores_theta = apply_along_axis2(pol.score,2,actions,features)
        scores_w = apply_along_axis2(pol.score_w,2,actions,features)
        disc_rewards = discount(rewards,gamma)

        cum_scores_theta = np.cumsum(scores_theta,1)
        cum_scores_w = np.cumsum(scores_w,1)

        cum_scores = cum_scores_w * cum_scores_theta

        estimates = np.sum((cum_scores.T * disc_rewards.T).T,1)
        return np.mean(estimates, 0) if average else estimates

    def gpomdp2(self, features, actions, rewards, gamma, pol, average=True):
        scores_theta = apply_along_axis2(pol.score,2,actions,features)
        scores_w = apply_along_axis2(pol.score_w,2,actions,features)
        disc_rewards = discount(rewards,gamma)

        cum_sum_theta = np.cumsum(scores_theta, 1)

        grad_prods = scores_w * cum_sum_theta
        cum_sum_grad_prods = np.cumsum(grad_prods, 1)

        estimates = np.sum((cum_sum_grad_prods.T * disc_rewards.T).T, 1)
        return np.mean(estimates, 0) if average else estimates

class GradMixedEstimator(object):
    """Estimates the mixed derivative ∇σ∇θJ(θ,σ)
    """
    def estimate(self, features, actions, rewards, gamma, pol, average=True):
        h_est = HEstimator('gpomdp2')
        h = h_est.estimate(features, actions, rewards, gamma, pol, average=average)

        gradtheta_est = Estimator('gpomdp')
        grad_theta = gradtheta_est.estimate(features, actions, rewards, gamma, pol, average=average)

        return h - 2 / (pol.sigma) * grad_theta


class GradDeltaWEstimator(object):
    def __init__(self, tp):
        self.tp = tp

        self.C1 = (1 - tp.gamma)**3 * math.sqrt(2 * math.pi)
        self.C2 = tp.gamma * math.sqrt(2 * math.pi) * tp.R * tp.M**2
        self.C3 = 2*(1 - tp.gamma) * tp.volume * tp.R * tp.M**2


    def estimate(self, features, actions, rewards, gamma, pol, average=True):
        sigma = pol.sigma

        c = pol.penaltyCoeff(self.tp.R, self.tp.M, self.tp.gamma, self.tp.volume)
        d = pol.penaltyCoeffSigma(self.tp.R, self.tp.M, self.tp.gamma, self.tp.volume)

        print('c:', c)
        print('d:', d)

        alphaStar=1/(2*c)

        print('alphaStar', alphaStar)

        grad_theta = Estimator('gpomdp').estimate(features, actions, rewards, gamma, pol, average=True)
        grad_mixed = GradMixedEstimator().estimate(features, actions, rewards, gamma, pol, average=True)

        print('grad_theta', grad_theta, 'grad_mixed', grad_mixed)

        grad_sigma_alpha_star = sigma**2 * (2*self.C1*self.C2*sigma + 3*self.C1*self.C3) / (pol.act_dim * (self.C2 * sigma + self.C3)**2)
        grad_sigma_norm_grad_theta = 2 * grad_theta * grad_mixed

        print('grad_sigma_alpha_star', grad_sigma_alpha_star, 'grad_sigma_norm_grad_theta', grad_sigma_norm_grad_theta)

        # Compute the gradient for sigma
        grad_local_step = (1/2) * grad_theta**2 * grad_sigma_alpha_star
        grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

        print('grad_local_step', grad_local_step, 'grad_far_sighted', grad_far_sighted)

        gradDelta = grad_local_step + grad_far_sighted
        gradDeltaW = gradDelta * math.exp(pol.w)

        print('gradDelta', gradDelta, 'gradDeltaW', gradDeltaW)

        return gradDeltaW

class Estimators(object):
    def __init__(self, tp):
        self.tp = tp

        self.C1 = (1 - tp.gamma)**3 * math.sqrt(2 * math.pi)
        self.C2 = tp.gamma * math.sqrt(2 * math.pi) * tp.R * tp.M**2
        self.C3 = 2*(1 - tp.gamma) * tp.volume * tp.R * tp.M**2

        self.gamma = tp.gamma


    def update(self, tp):
        self.__init__(tp)

    def __compute_baseline_theta_gpomdp(self, cum_scores, disc_rewards):
        den = np.mean(cum_scores**2,0)
        np.putmask(den,den==0,1)
        return np.mean(((cum_scores**2).T*disc_rewards.T).T,0)/den

    def estimate(self, features, actions, rewards, pol, average=True, use_baseline=True):
        #Data
        assert features.shape[:2]==actions.shape[:2]==rewards.shape[:2]
        N = features.shape[0]
        H = features.shape[1]
        m = features.shape[2] if len(features.shape)>2 else 1

        #Q function
        disc_rewards = discount(rewards,self.gamma)

        #Eligibility vector
        scores_theta = apply_along_axis2(pol.score,2,actions,features)
        scores_sigma = apply_along_axis2(pol.score_sigma,2,actions,features)
        scores_w = scores_sigma * math.exp(pol.w)

        cum_scores_theta = np.cumsum(scores_theta,1)
        cum_scores_w = np.cumsum(scores_w,1)

        #Optimal baseline:
        if use_baseline and N>1:
            b_theta = self.__compute_baseline_theta_gpomdp(cum_scores_theta, disc_rewards)
            b_w = self.__compute_baseline_theta_gpomdp(cum_scores_w, disc_rewards)
        else:
            b = np.zeros((H,m))
            b_w = np.zeros((H,m))

        #gradient estimate:
        estimates_theta = np.sum((cum_scores_theta.T*disc_rewards.T).T - cum_scores_theta*b_theta,1)
        estimates_w = np.sum((cum_scores_w.T*disc_rewards.T).T - cum_scores_w*b_w,1)

        grad_theta = np.mean(estimates_theta,0)
        grad_w = np.mean(estimates_w,0)

        # ESTIMATES H

        grad_prods = scores_sigma * cum_scores_theta
        cum_sum_grad_prods = np.cumsum(grad_prods, 1)

        estimates_h = np.sum((cum_sum_grad_prods.T * disc_rewards.T).T, 1)
        h = np.mean(estimates_h, 0)

        grad_mixed = h - 2 / (pol.sigma) * grad_theta

        sigma = pol.sigma

        c = pol.penaltyCoeff(self.tp.R, self.tp.M, self.tp.gamma, self.tp.volume)
        d = pol.penaltyCoeffSigma(self.tp.R, self.tp.M, self.tp.gamma, self.tp.volume)

        alphaStar=1/(2*c)
        # ESTIMATES GRAD DELTA W
        grad_sigma_alpha_star = sigma**2 * (2*self.C1*self.C2*sigma + 3*self.C1*self.C3) / (pol.act_dim * (self.C2 * sigma + self.C3)**2)
        grad_sigma_norm_grad_theta = 2 * grad_theta * grad_mixed
        grad_local_step = (1/2) * grad_theta**2 * grad_sigma_alpha_star
        grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

        gradDelta = grad_local_step + grad_far_sighted
        gradDeltaW = gradDelta * math.exp(pol.w)

        # @TOCHANGE
        # gradDeltaW = 1

        return {'grad_theta' : grad_theta, 'grad_w' : grad_w, 'grad_mixed' : grad_mixed, 'gradDeltaW' : gradDeltaW}




def performance(rewards,gamma=None,average=True):
    discounted = (gamma!=None)
    if discounted:
        Js = np.sum(discount(rewards,gamma),1)
    else:
        Js = np.sum(rewards,1)

    return np.mean(Js) if average else Js

def discount(rewards,gamma):
    #Applies the discount factor to rewards
    N = rewards.shape[0]
    H = rewards.shape[1]
    discounts = gamma**np.indices((N,H))[1]
    return rewards*discounts
