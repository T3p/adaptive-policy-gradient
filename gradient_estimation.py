import numpy as np
from utils import *
import numba
import scipy.stats

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

    # def gpomdp_w(self,features,actions,rewards,gamma,pol,use_baseline=True,average=True):
    #     """Batch G(PO)MDP policy gradient estimator with variance-minimizing baseline
    #
    #     Parameters:
    #     features -- N x H x m array containing feature vectors for N episodes of length H
    #     actions -- N x H x d array containing actions for N episodes of length H
    #     rewards -- N x H array containing rewards for N episodes of length H
    #     gamma -- discount factor
    #     use_baseline -- if False, a baseline of b=0 is used instead
    #
    #     Returns:
    #     the averaged gradient estimate if average==True, an array containing the N estimates otherwise
    #     """
    #     #Data
    #     assert features.shape[:2]==actions.shape[:2]==rewards.shape[:2]
    #     N = features.shape[0]
    #     H = features.shape[1]
    #     m = features.shape[2] if len(features.shape)>2 else 1
    #
    #     #Q function
    #     disc_rewards = discount(rewards,gamma)
    #
    #     #Eligibility vector
    #     scores = apply_along_axis2(pol.score_w,2,actions,features)
    #     cum_scores = np.cumsum(scores,1)
    #
    #     #Optimal baseline:
    #     b = np.zeros((H,m))
    #     if use_baseline and N>1:
    #         den = np.mean(cum_scores**2,0)
    #         np.putmask(den,den==0,1)
    #         b = np.mean(((cum_scores**2).T*disc_rewards.T).T,0)/den
    #
    #     #gradient estimate:
    #     estimates =  np.sum((cum_scores.T*disc_rewards.T).T - cum_scores*b,1)
    #     return np.mean(estimates,0) if average else estimates
    #




class Estimators(object):
    def __init__(self, tp, opt_constr = None):
        self.tp = tp
        self.opt_constr = opt_constr

        if (opt_constr is not None) and (opt_constr.approximate_gradients == True):
            self.approximate = True
        else:
            self.approximate = False

        self.C1 = (1 - tp.gamma)**3 * math.sqrt(2 * math.pi)
        self.C2 = tp.gamma * math.sqrt(2 * math.pi) * tp.R * tp.M**2
        self.C3 = 2*(1 - tp.gamma) * tp.volume * tp.R * tp.M**2

        self.gamma = tp.gamma


    def update(self, tp):
        self.__init__(tp, self.opt_constr)

    def __compute_baseline_theta_gpomdp(self, cum_scores, disc_rewards):
        den = np.mean(cum_scores**2,0)
        np.putmask(den,den==0,1)
        return np.mean(((cum_scores**2).T*disc_rewards.T).T,0)/den

    def __compute_baseline_h(self, cum_scores_theta, cum_scores_sigma, disc_rewards):
        num = np.mean(cum_scores_sigma * cum_scores_theta * disc_rewards * (cum_scores_sigma + cum_scores_theta), 0)
        den = np.mean((cum_scores_sigma + cum_scores_theta)**2, 0)
        np.putmask(den, den==0, 1)
        return (cum_scores_sigma + cum_scores_theta) / (cum_scores_sigma * cum_scores_theta) * num/den

    def estimate_full(self, features, actions, rewards, pol, average=True, use_baseline=True):
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
        if use_baseline and N>=1:
            b_theta = self.__compute_baseline_theta_gpomdp(cum_scores_theta, disc_rewards)
            b_w = self.__compute_baseline_theta_gpomdp(cum_scores_w, disc_rewards)
        else:
            b_theta = np.zeros((H,m))
            b_w = np.zeros((H,m))

        #gradient estimate:
        estimates_theta = np.sum((cum_scores_theta.T*disc_rewards.T).T - cum_scores_theta*b_theta,1)
        estimates_w = np.sum((cum_scores_w.T*disc_rewards.T).T - cum_scores_w*b_w,1)

        grad_theta = np.mean(estimates_theta,0)
        grad_w = np.mean(estimates_w,0)

        # ESTIMATES H



        # ESTIMATOR 1 (REINFORCE)
        Q_estimate = np.sum(disc_rewards, 1)
        sum_theta_estimate = np.sum(scores_theta, 1)
        sum_sigma_estimate = np.sum(scores_sigma, 1)

        estimates_h_reinforce = Q_estimate * sum_theta_estimate * sum_sigma_estimate
        h_reinforce = np.mean(estimates_h_reinforce, 0)

        # ESTIMATOR 2 (FIRST STEP GPOMDP)
        cum_scores_sigma = np.cumsum(scores_sigma,1)
        cum_scores_sigma_theta = cum_scores_sigma * cum_scores_theta
        estimates_h_gpomdp1 = np.sum((cum_scores_sigma_theta.T*disc_rewards.T).T, 1)
        h_gpomdp1 = np.mean(estimates_h_gpomdp1, 0)

        # ESTIMATOR 2 WITH BASELINE
        num = np.mean(cum_scores_sigma * cum_scores_theta * disc_rewards * (cum_scores_sigma + cum_scores_theta), 0)
        den = np.mean((cum_scores_sigma + cum_scores_theta)**2, 0)
        np.putmask(den, den==0, 1)
        baseline_gpomdp1_tilde = num/den

        baseline_gpomdp1 = (cum_scores_sigma + cum_scores_theta) / (cum_scores_sigma * cum_scores_theta) * baseline_gpomdp1_tilde

        estimates_h_gpomdp1_baseline = np.sum((cum_scores_sigma_theta.T*disc_rewards.T).T - cum_scores_sigma_theta*baseline_gpomdp1,1)
        # estimates_h_gpomdp1_baseline = np.sum((cum_scores_sigma_theta.T*disc_rewards.T).T - cum_scores_sigma_theta*baseline_gpomdp1,1)
        h_gpomdp1_baseline = np.mean(estimates_h_gpomdp1_baseline, 0)


        # ESTIMATOR 3
        grad_prods = scores_sigma * cum_scores_theta
        cum_sum_grad_prods = np.cumsum(grad_prods, 1)
        estimates_h_gpomdp2 = np.sum((cum_sum_grad_prods.T * disc_rewards.T).T, 1)
        h_gpomdp2 = np.mean(estimates_h_gpomdp2, 0)


        def _compute_grad_mixed_deltaw(h_estimate):
            grad_mixed = h_estimate - 2 / (pol.sigma) * grad_theta

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

            return grad_mixed, gradDeltaW

        def _compute_grad_exact():
            if 'env' not in self.__dict__:
                import lqg1d
                self.env = lqg1d.LQG1D()

            sigma = pol.sigma

            M = self.env.max_pos
            ENV_GAMMA = self.env.gamma
            ENV_VOLUME = 2*self.env.max_action
            ENV_R = np.asscalar(self.env.R)
            ENV_Q = np.asscalar(self.env.Q)
            ENV_B = np.asscalar(self.env.B)
            ENV_MAX_ACTION = self.env.max_action

            MAX_REWARD = ENV_Q * M**2 + ENV_R * ENV_MAX_ACTION**2

            C1 = (1 - ENV_GAMMA)**3 * math.sqrt(2 * math.pi)
            C2 = ENV_GAMMA * math.sqrt(2 * math.pi) * MAX_REWARD * M**2
            C3 = 2*(1 - ENV_GAMMA) * ENV_VOLUME * MAX_REWARD * M**2

            m = 1

            # c = utils.computeLoss(MAX_REWARD, M, ENV_GAMMA, ENV_VOLUME, sigma)
            c = pol.penaltyCoeff(MAX_REWARD, M, ENV_GAMMA, ENV_VOLUME)
            # d = utils.computeLossSigma(MAX_REWARD, M, ENV_GAMMA, ENV_VOLUME, sigma)

            alphaStar=1/(2*c)

            gradK = self.env.grad_K(np.asscalar(pol.theta_mat), sigma)
            gradMixed = self.env.grad_mixed(np.asscalar(pol.theta_mat), sigma)

            grad_sigma_alpha_star = sigma**2 * (2*C1*C2*sigma + 3*C1*C3) / (m * (C2 * sigma + C3)**2)
            grad_sigma_norm_grad_theta = 2 * gradK * gradMixed

            # Compute the gradient for sigma
            grad_local_step = (1/2) * gradK**2 * grad_sigma_alpha_star
            grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

            gradDelta = grad_local_step + grad_far_sighted
            gradDeltaW = gradDelta * math.exp(pol.w)

            return gradMixed, gradDeltaW


        grad_mixed_reinforce, grad_delta_reinforce = _compute_grad_mixed_deltaw(h_reinforce)
        grad_mixed_gpomdp1, grad_delta_gpomdp1 = _compute_grad_mixed_deltaw(h_gpomdp1)
        grad_mixed_gpomdp2, grad_delta_gpomdp2 = _compute_grad_mixed_deltaw(h_gpomdp2)
        grad_mixed_exact, grad_delta_exact = _compute_grad_exact()
        grad_mixed_gpomdp1_baseline, grad_delta_gpomdp1_baseline = _compute_grad_mixed_deltaw(h_gpomdp1_baseline)

        grad_mixed = grad_mixed_gpomdp1_baseline
        grad_deltaW = grad_delta_gpomdp1_baseline

        return {'grad_theta' : grad_theta,
                'grad_w' : grad_w,
                'grad_mixed' : grad_mixed,
                'gradDeltaW' : grad_deltaW,
                'grad_mixed_reinforce' : grad_mixed_reinforce,
                'grad_delta_reinforce' : grad_delta_reinforce,
                'grad_mixed_gpomdp1' : grad_mixed_gpomdp1,
                'grad_delta_gpomdp1' : grad_delta_gpomdp1,
                'grad_mixed_gpomdp2' : grad_mixed_gpomdp2,
                'grad_delta_gpomdp2' : grad_delta_gpomdp2,
                'grad_mixed_exact' : grad_mixed_exact,
                'grad_delta_exact' : grad_delta_exact,
                'grad_mixed_gpomdp1_baseline' : grad_mixed_gpomdp1_baseline,
                'grad_delta_gpomdp1_baseline' : grad_delta_gpomdp1_baseline}

    def _estimate(self, features, actions, rewards, pol, average=True, use_baseline=True):
        return {'grad_theta' : -100,
                'grad_w' : -200,
                'grad_mixed' : -50,
                'gradDeltaW' : 0.01,
                'grad_theta_low' : -100,
                'grad_theta_high' : -100,
                'grad_w_low' : -200,
                'grad_w_high' : -200}

    def estimate(self, features, actions, rewards, scores_theta, scores_sigma, trace_lengths, pol, average=True, use_baseline=True):
        #Data
        assert features.shape[:2]==actions.shape[:2]==rewards.shape[:2]
        N = features.shape[0]
        H = features.shape[1]
        m = features.shape[2] if len(features.shape)>2 else 1

        #Q function
        disc_rewards = discount(rewards,self.gamma)

        #Eligibility vector
        # scores_theta = apply_along_axis2(pol.score,2,actions,features)
        # scores_sigma = apply_along_axis2(pol.score_sigma,2,actions,features)
        scores_w = scores_sigma * math.exp(pol.w)

        cum_scores_theta = np.cumsum(scores_theta,1)
        cum_scores_sigma = np.cumsum(scores_sigma,1)
        cum_scores_w = np.cumsum(scores_w,1)


        # MASK CUMULATIVE SCORES

        idxs_theta = np.indices(cum_scores_theta.shape)[1]
        idxs_w = np.indices(cum_scores_w.shape)[1]

        if np.min(trace_lengths) != H:

            cum_scores_theta = np.ma.array(cum_scores_theta, mask=idxs_theta > trace_lengths.reshape((-1, 1, 1)))# if pol.feat_dim>1 else (-1, 1)))
            cum_scores_w = np.ma.array(cum_scores_w, mask=idxs_w > trace_lengths.reshape(-1, 1))
            cum_scores_sigma = np.ma.array(cum_scores_sigma, mask=idxs_w > trace_lengths.reshape(-1, 1))
            disc_rewards = np.ma.array(disc_rewards, mask=np.indices(disc_rewards.shape)[1] > trace_lengths.reshape(-1, 1))
            
        #Optimal baseline:
        if use_baseline and N>=1:
            b_theta = _compute_baseline_theta_gpomdp(cum_scores_theta, disc_rewards)
            b_w = _compute_baseline_theta_gpomdp(cum_scores_w, disc_rewards)
            b_h = _compute_baseline_h(cum_scores_theta, cum_scores_sigma, disc_rewards)
        else:
            b_theta = np.zeros((H,m))
            b_w = np.zeros((H,))    #@TODO Change this when using multiple actions
            b_h = np.zeros((N,H,m))

        #gradient estimate:
        estimates_theta = np.sum((cum_scores_theta.T*disc_rewards.T).T - cum_scores_theta*b_theta,1)
        estimates_w = np.sum((cum_scores_w.T*disc_rewards.T).T - cum_scores_w*b_w,1)

        grad_theta = np.mean(estimates_theta,0)
        grad_w = np.mean(estimates_w,0)


        # ESTIMATES H

        cum_scores_sigma_theta = (cum_scores_sigma.T * cum_scores_theta.T).T
        estimates_h = np.sum((cum_scores_sigma_theta.T*disc_rewards.T).T - cum_scores_sigma_theta*b_h,1)

        # print('H STD: ', np.std(estimates_h, axis=0))

        h_gpomdp = np.mean(estimates_h, 0)

        # print('h_gpomdp = ', h_gpomdp)
        # print('cum_scores_sigma_theta: ', cum_scores_sigma_theta.shape, 'b_h shape: ', b_h.shape)


        def _compute_grad_mixed_deltaw(h_estimate):
            grad_mixed = h_estimate - 2 / (pol.sigma) * grad_theta

            sigma = pol.sigma

            c = pol.penaltyCoeff(self.tp.R, self.tp.M, self.tp.gamma, self.tp.volume)

            alphaStar=1/(2*c)
            # ESTIMATES GRAD DELTA W
            grad_sigma_alpha_star = sigma**2 * (2*self.C1*self.C2*sigma + 3*self.C1*self.C3) / (pol.act_dim * (self.C2 * sigma + self.C3)**2)
            grad_sigma_norm_grad_theta = 2 * np.dot(grad_theta, grad_mixed)
            grad_local_step = (1/2) * np.linalg.norm(grad_theta.ravel(), 2)**2 * grad_sigma_alpha_star
            grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

            gradDelta = grad_local_step + grad_far_sighted
            gradDeltaW = gradDelta * math.exp(pol.w)

            return grad_mixed, gradDeltaW


        grad_mixed, grad_deltaW = _compute_grad_mixed_deltaw(h_gpomdp)

        # print('GRAD_DELTAW = ', grad_deltaW)
        # print('GRADW = ', grad_w)
        # print('GRAD_THETA: ', grad_theta)


        if self.approximate == True:
            N = self.opt_constr.N_min
            delta = self.opt_constr.delta
            eps_theta = np.sqrt(np.var(estimates_theta, ddof=1) / N)*scipy.stats.t.ppf(1 - (delta/2), N-1)
            eps_w = np.sqrt(np.var(estimates_w, ddof=1) / N) * scipy.stats.t.ppf(1 - (delta/2), N-1)
        else:
            eps_theta = 0
            eps_w = 0

        #print('GRAD_THETA', grad_theta, 'GRAD_W', grad_w, 'GRAD_MIXED', grad_mixed, 'GRAD_DELTA', grad_deltaW)


        return {'grad_theta' : grad_theta,
                'grad_w' : grad_w,
                'grad_mixed' : grad_mixed,
                'gradDeltaW' : grad_deltaW,
                'grad_theta_low' : np.linalg.norm(np.max(np.abs(grad_theta) - eps_theta, 0).ravel(), 2), # L2 norm
                'grad_theta_high' : np.linalg.norm((np.abs(grad_theta) + eps_theta).ravel(), 1),      # L1 norm
                'grad_w_low' : np.linalg.norm(np.max(np.abs(grad_w) - eps_w, 0).ravel(), 2),        # L2 norm
                'grad_w_high' : np.linalg.norm((np.abs(grad_w) + eps_w).ravel(), 1),              # L1 norm
                'grad_theta_inf_low' : np.linalg.norm(np.max(np.abs(grad_theta) - eps_theta, 0).ravel(), np.inf),
                'grad_theta_inf_high' : np.linalg.norm((np.abs(grad_theta) + eps_theta).ravel(), np.inf)}


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

@numba.jit
def _compute_baseline_theta_gpomdp(cum_scores, disc_rewards):
    den = np.mean(cum_scores**2,0)
    np.putmask(den,den==0,1)
    return np.mean(((cum_scores**2).T*disc_rewards.T).T,0)/den

@numba.jit
def _compute_baseline_h(cum_scores_theta, cum_scores_sigma, disc_rewards):
    sum_theta_sigma = (cum_scores_sigma.T + cum_scores_theta.T).T
    mul_theta_sigma = (cum_scores_sigma.T * cum_scores_theta.T).T

    num = np.mean((mul_theta_sigma.T * disc_rewards.T).T * sum_theta_sigma, 0)
    den = np.mean(sum_theta_sigma**2, 0)
    np.putmask(den, den==0, 1)
    return sum_theta_sigma / mul_theta_sigma * num/den
