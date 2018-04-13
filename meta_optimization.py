import math
from utils import *
from gradient_estimation import Estimator

class TaskProp:
    """Properties of the RL task, true or estimated from experience"""

    def __init__(self,gamma,H,min_action,max_action,R=None,M=None,min_state=None,max_state=None,volume=None,diameter=None):
        """Parameters:
            R -- max absolute-value reward
            M -- upper bound on all state features
            gamma -- discount factor
            H -- episode length
            min_state --
            max_state --
            min_action --
            max_action --
            volume -- volume of the action space
            diameter -- maximum euclidean distance among possible actions
        """
        self.gamma = gamma
        self.H = H
        self.min_action = np.atleast_1d(min_action)
        self.max_action = np.atleast_1d(max_action)

        self.R = 0 if R==None else R
        self.M = 0 if M==None else M
        self.diameter = 0 if diameter==None else diameter
        self.min_state = np.inf if min_state==None else np.atleast_1d(min_state)
        self.max_state = -np.inf if max_state==None else np.atleast_1d(max_state)
        self.volume = 0 if volume==None else volume
        self.diameter = 0 if diameter==None else diameter

        self.min_action_seen = np.inf
        self.max_action_seen = -np.inf

    def update(self,features,actions,rewards,local=False):
        R = np.amax(abs(rewards))
        max_state = np.amax(features,(0,1))
        min_state = np.amin(features,(0,1))
        M = np.amax(abs(features))
        self.R = R if local else max(self.R,R)
        self.max_state = max_state if local else np.maximum(self.max_state,max_state)
        self.min_state = min_state if local else np.minimum(self.min_state,min_state)
        self.M = M if local else max(self.M,M)

        min_action_seen = np.amin(actions,(0,1))
        max_action_seen = np.amax(actions,(0,1))
        self.min_action_seen = min_action_seen if local else np.minimum(self.min_action_seen,min_action_seen)
        self.max_action_seen = max_action_seen if local else np.maximum(self.max_action_seen,max_action_seen)
        volume = np.multiply.reduce(self.max_action_seen - self.min_action_seen) #<- box
        diameter = np.linalg.norm(np.atleast_1d(self.max_action_seen) - np.atleast_1d(self.min_action_seen),2)
        self.volume = max(self.volume,volume)
        self.diameter = max(self.diameter,diameter)

        #print R,M,volume

    @staticmethod
    def fromLQGEnvironment(env):
        return TaskProp(
            env.gamma,
            env.horizon,
            -env.max_action,
            env.max_action,
            R=np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2),
            M=env.max_pos,
            min_state=-env.max_pos,
            max_state=env.max_pos,
            volume=2*env.max_action,
            diameter=None)


class GradStats:
    """Statistics about a gradient estimate"""

    def __init__(self,grad_samples):
        """Parameters:
            grad_samples: gradient estimates
        """
        self.grad = np.mean(grad_samples,0)
        self.max_grad = np.max(abs(self.grad))
        self.k_max = np.argmax(abs(self.grad))
        self.grad_samples = grad_samples[:,self.k_max] if len(grad_samples.shape)>1 else grad_samples
        self.sample_range = self.sample_var = None

    def get_estimate(self):
        return self.grad

    def get_max(self):
        return self.max_grad

    def get_amax(self):
        return self.k_max

    def get_var(self):
        if self.sample_var==None:
            self.sample_var = np.var(self.grad_samples,ddof=1)
        return self.sample_var

    def get_range(self):
        if self.sample_range==None:
            self.sample_range = max(self.grad_samples) - min(self.grad_samples)
        return self.sample_range


class OptConstr:
    """Constraints on the meta-optimization process"""

    def __init__(self,delta=0.95,N_min=2,N_max=999999,N_tot=30000000,max_iter=30000000, approximate_gradients=True):
        """Parameters:
            delta : maximum allowed worsening probability
            N_min : min allowed batch size
            N_max : max allowed batch size
            N_tot : total number of possible trajectories
        """
        self.approximate_gradients = approximate_gradients
        self.delta = delta
        self.N_min = N_min
        self.N_max = N_max
        self.N_tot = N_tot
        self.max_iter = max_iter

#Default constraints
default_constr = OptConstr()


def gradRange(pol,tp):
    """Range of the gradient estimate

        Parameters:
        pol -- policy in use
        tp -- TaskProp object containing the (true or last estimated) properties of the task
    """
    Q_sup = float(tp.volume*tp.R)/(1-tp.gamma)
    return float(tp.M*tp.diameter*Q_sup)/pol.sigma**2

def estError(d,f,N):
    """Generic estimation error bound

        Parameters:
        d -- 1/sqrt(N) coefficient
        f -- 1/N coefficient
        N -- batch size
    """
    return float(d)/math.sqrt(N) + float(f)/N


class MetaSelector(object):
    """Meta-parameters for the policy gradient problem"""
    def __init__(self,alpha,N):
        self.alpha = np.atleast_1d(alpha)
        self.N = N

    def select(self,pol,gs,tp,N_pre,iteration):
        return self.alpha,self.N,False

ConstMeta = MetaSelector

class BudgetMetaSelector(object):
    def __init__(self):
        pass

    def select_alpha(self, policy, gradients, tp, N1, iteration, budget=None):
        """Perform a safe update on theta
        """
        sigma = policy.sigma
        c = policy.penaltyCoeff(tp.R, tp.M, tp.gamma, tp.volume)

        if budget is None:  # Safe step
            return gradients['grad_theta_low']**2/(2*c*gradients['grad_theta_high']**2),N1,True

        # if budget / N1 >= -(gradients['grad_theta']**2)/(4*c):
        if budget/N1 >= -(gradients['grad_theta_low']**4) / (4*c*gradients['grad_theta_high']**2):
            # alpha_star = (1 + math.sqrt(1 - (4 * c * (-budget / N1))/(gradients['grad_theta']**2))) / (2 * c)

            alpha_star = (gradients['grad_theta_low']**2 + math.sqrt(gradients['grad_theta_low']**4 + 4*c*(budget/N1)*gradients['grad_theta_high']**2)) / (2*c*gradients['grad_theta_high']**2)
        else:
            # alpha_star = 1/(2*c)
            alpha_star = gradients['grad_theta_low']**2/(2*c*gradients['grad_theta_high']**2)

        return alpha_star,N1,False

    def select_beta(self, policy, gradients, tp, N3, iteration, budget):
        sigma = policy.sigma

        d = policy.penaltyCoeffSigma(tp.R, tp.M, tp.gamma, tp.volume)

        if budget is None:  # Safe step
            return gradients['grad_w_low']**2/(2*d*gradients['grad_w_high']**2),N3,True

        # assert that the budget is small enough
        #if budget / N3 >= -(gradients['grad_w']**2)/(4*d):
        if budget/N3 >= -(gradients['grad_w_low']**4) / (4*d*gradients['grad_w_high']**2):
            # beta_tilde_minus = (1 - math.sqrt(1 - (4 * d * (-budget/N3))/(gradients['grad_w']**2))) / (2 * d)
            # beta_tilde_plus = (1 + math.sqrt(1 - (4 * d * (-budget/N3))/(gradients['grad_w']**2))) / (2 * d)

            beta_tilde_plus = (gradients['grad_w_low']**2 + math.sqrt(gradients['grad_w_low']**4 + 4*d*(budget/N3)*gradients['grad_w_high']**2)) / (2*d*gradients['grad_w_high']**2)
            beta_tilde_minus = (gradients['grad_w_low']**2 - math.sqrt(gradients['grad_w_low']**4 + 4*d*(budget/N3)*gradients['grad_w_high']**2)) / (2*d*gradients['grad_w_high']**2)

            if gradients['gradDeltaW'] / gradients['grad_w'] >= 0:
                beta_star = beta_tilde_plus * gradients['grad_w'] / gradients['gradDeltaW']
            else:
                beta_star = beta_tilde_minus * gradients['grad_w'] / gradients['gradDeltaW']

        else:
            beta_star = gradients['grad_w_low']**2/(2*d*gradients['grad_w_high']**2) * gradients['grad_w'] / gradients['gradDeltaW']
            # beta_star = 1/(2*d) * gradients['grad_w'] / gradients['gradDeltaW']



        return beta_star,N3,False

class VanishingMeta(MetaSelector):
    def __init__(self,alpha,N,alpha_exp=0.5,N_exp = 0):
        super(VanishingMeta,self).__init__(alpha,N)
        self.alpha_exp = alpha_exp
        self.N_exp = N_exp

    def select(self,pol,gs,tp,N_pre,iteration):
        return self.alpha/(iteration**self.alpha_exp),self.N*iteration**self.N_exp,False

class MetaOptimizer(MetaSelector):
    """Tool to compute the optimal meta-parameters for a policy gradient problem"""

    def __init__(self,bound_name='bernstein',constr=default_constr,estimator_name='gpomdp',samp=True,cost_sensitive_step=False,c=None):


        bounds = {'chebyshev': self.__chebyshev, 'hoeffding': self.__hoeffding, 'bernstein': self.__bernstein}

        self.bound_name = bound_name
        self.bound = bounds[bound_name]
        self.constr = constr
        self.estimator_name = estimator_name
        self.samp = samp
        self.cs_step = cost_sensitive_step
        self.c = c

    def alphaStar(self,pol,tp):
        """Optimal step size for the adaBatch algorithm when the corresponding optimal
            batch size is used

            Parameters:
            pol -- policy in use
            tp -- TaskProp object containing the (true or last estimated) properties of the task
       """

        c = pol.penaltyCoeff(tp.R,tp.M,tp.gamma,tp.volume,self.c)
        return (13-3*math.sqrt(17))/(4*c)

    def alphaPost(self,pol,tp,max_grad,eps):
        """Optimal step size given an upper bound of the estimaton error,
            depending on the batch size that is actually used

            Parameters:
            pol -- policy in use
            tp -- TaskProp object containing the (true or last estimated) properties of the task
            gs -- GradStats object containing statistics on the last gradient estimate
        """
        c = pol.penaltyCoeff(tp.R,tp.M,tp.gamma,tp.volume,self.c)
        #print 'c = ', c
        return (max_grad - eps)**2/(2*c*(max_grad + eps)**2)


    def select(self,pol,gs,tp,N_pre,iteration):
        """Compute optimal step size and batch size

            Parameters:
            pol -- policy in use
            gs -- GradStats object containing statistics about last gradient estimate
            tp -- TaskProp object containing the (true or last estimated) properties of the task
            N_pre -- batch size that was actually used to compute the last gradient estimate

            Returns:
            alpha -- the optimal non-scalar step size
            N -- the optimal batch size
            unsafe -- true iff no improvement can be guaranteed at all
        """
        d,f,eps_star,N_star = self.bound(pol,gs,tp)
        actual_eps = estError(d,f,N_pre)

        alpha_k = self.alphaPost(pol,tp,gs.get_max(),actual_eps)

        if(self.cs_step):
            alpha_k = 2*alpha_k

        N = min(self.constr.N_max,max(self.constr.N_min,N_star))
        safe = eps_star<gs.get_max()

        alpha = np.zeros(pol.param_len)
        alpha[gs.get_amax()] = alpha_k

        return alpha,N,safe

    def __str__(self):
        return 'Estimator: {}, Bound: {}, Empirical range: {}, delta = {}'.format(self.estimator_name,self.bound_name,self.samp,self.constr.delta)

    def __closedOpt(self,d,max_grad):
        #Generic closed form optimization for N and corresponding estimation error

        eps_star = 0.25*(math.sqrt(17) - 3)*max_grad
        N_star = int(math.ceil(float(d**2)/eps_star**2))
        return eps_star,N_star


    def __chebyshev(self,pol,gs,tp):
        #Batch size optimizer using Chebyshev's bound
        if self.estimator_name=='reinforce':
            d =  math.sqrt((tp.R**2*tp.M**2*tp.H*(1-tp.gamma**tp.H)**2)/ \
                    (pol.sigma**2*(1-tp.gamma)**2*self.constr.delta))
        elif self.estimator_name=='gpomdp':
            d = math.sqrt((tp.R**2*tp.M**2)/(self.constr.delta*pol.sigma**2*(1-tp.gamma)**2) * \
                           ((1-tp.gamma**(2*tp.H))/(1-tp.gamma**2)+ tp.H*tp.gamma**(2*tp.H)  - \
                                2 * tp.gamma**tp.H  * (1-tp.gamma**tp.H)/(1-tp.gamma)))
        else:
            assert False

        return (d,0) + self.__closedOpt(d,gs.get_max())

    def __hoeffding(self,pol,gs,tp):
        #Batch size optimizer using Hoeffding's bound
        if self.samp:
            rng = gs.get_range()
        else:
            rng = gradRange(pol,tp)

        d = rng*math.sqrt(math.log(2./self.constr.delta)/2)
        return (d,0) + self.__closedOpt(d,gs.get_max())


    def __evaluateN(self,N,d,f,c,max_grad):
        #Objective function Upsilon for batch size N
        eps = float(d)/math.sqrt(N) + float(f)/N
        upsilon = (max_grad - eps)**4/ \
                    (4*c*(max_grad + eps)**2*N)
        return upsilon,eps

    def __bernstein(self,pol,gs,tp):
        #Batch size optimizer using an empirical Bernstein's bound (Mnih et al., 2008)
        if self.samp:
            rng = gs.get_range()
        else:
            rng = gradRange(pol,tp)

        c = pol.penaltyCoeff(tp.R,tp.M,tp.gamma,tp.volume,self.c)
        d = math.sqrt(2*math.log(3.0/self.constr.delta)*gs.get_var())
        f = 3*rng*math.log(3.0/self.constr.delta)

        N_0 = min(self.constr.N_max,max(self.constr.N_min,int(((d + math.sqrt(d**2 + 4*f*gs.get_max())) \
                /(2*gs.get_max()))**2) + 1))
        ups_max = -np.inf
        eps_star = np.inf
        N_star = N_0
        #for n in range(N_0,self.constr.N_max):
        #    ups,eps = self.__evaluateN(n,d,f,c,gs.get_max())
        #    if ups>ups_max:
        #        ups_max = ups
        #        eps_star = eps
        #        N_star = n
        #    else:
        #        break

        return d,f,eps_star,N_star
