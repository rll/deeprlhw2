"""

Proximal policy optimization

It's like "Trust Region Policy Optimization" from [1]

The pseudocode is as follows:

1. Collect trajectories using current policy
2. maximize the following objective with respect to policy parameters

    average_n [ pi(a_n | s_n) / piold(a_n | s_n) * A_n 
     - lambda * KL( piold( . | s_n) \\ pi( . | s_n) ) ]

  i.e., maximize the surrogate loss function with a penalty on KL divergence between
  old ad new policy.



[1] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). 
Trust Region Policy Optimization. http://arxiv.org/abs/1502.05477

"""


import numpy as np
from hw_utils import Message, discount, explained_variance_1d
from collections import OrderedDict
import time, multiprocessing, itertools, sys
from lbfgs import lbfgs
from rl import rollout, pathlength, NoValueFunction, Policy
import scipy.optimize

class PPOPolicy(Policy):
    """
    A Policy with some extra functions needed by PPO algorithm
    """

    def step(self, states, actions):
        raise NotImplementedError

    def get_parameters_flat(self):
        raise NotImplementedError
    def set_parameters_flat(self):
        raise NotImplementedError
    def compute_surr_kl(self):
        raise NotImplementedError
    def compute_grad_lagrangian(self):
        raise NotImplementedError


class Globals: #pylint: disable=W0232
    pass

def rollout1(seed):
    np.random.seed(seed)
    return rollout(Globals.mdp, Globals.policy, Globals.max_pathlength)                


def run_ppo(mdp, policy,
        gamma,
        max_pathlength,
        timesteps_per_batch,
        n_iter,
        vf = None,
        lam=1.0,
        penalty_coeff=1.0,
        parallel = True,
        max_kl = 0.1
    ):
    """
    mdp : instance of MDP 
    policy : instance of PPOPolicy
    max_pathlength : maximum episode length (number of timesteps)
    n_iter : number of batches of PPO
    vf : instance of ValueFunction
    lam : lambda parameter of lambda for computing advantage estimator adv_t = delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots
           as described in http://arxiv.org/abs/1506.05254
    penalty_coeff : each iteration we solve the unconstrained minimization problem minimize_{theta} L(theta) + penalty_coeff * KL( thetaold, theta )
    parallel : use python's multiprocessing to parallelize the rollouts
    max_kl : hard limit on KL divergence between old and new policy for one iteration of PPO
    """

    assert isinstance(policy, PPOPolicy)

    if vf is None: vf = NoValueFunction()

    theta = policy.get_parameters_flat()

    seed_iter = itertools.count()
    start_time = time.time()

    numeptotal = 0

    for i in xrange(n_iter):
        print "********** Iteration %i ************"%i
        with Message("Generating paths"):
            total_ts = 0
            paths = []
            

            if parallel:
                # DO ROLLOUTS IN PARALLEL
                nproc = multiprocessing.cpu_count()
                if sys.platform == "darwin": nproc /= 2 
                # hyperthreading makes num cpu look twice as high
                # but there's no speedup
                # store data in global variables so it's accessible from forked processes
                # (which start when multiprocessing.Pool is created)
                Globals.mdp = mdp
                Globals.policy = policy
                Globals.max_pathlength = max_pathlength
                pool = multiprocessing.Pool(nproc)
                pending = []
                done = False
                while True:                    
                    if len(pending) < nproc and not done:                    
                        pending.append(pool.apply_async(rollout1, (seed_iter.next(),)))
                    stillpending = []
                    for job in pending:
                        if job.ready():
                            path = job.get()
                            paths.append(path)
                            total_ts += pathlength(path)
                        else:
                            stillpending.append(job)
                    pending = stillpending                
                    if total_ts > timesteps_per_batch: 
                        done = True
                        if len(pending) == 0:
                            break
                    time.sleep(.001)
                pool.close()
            else:
                # EQUIVALENT SERIAL CODE
                while True:
                    path = rollout(mdp, policy, max_pathlength)                
                    paths.append(path)
                    total_ts += pathlength(path)
                    if total_ts > timesteps_per_batch: 
                        break            


        with Message("Computing returns and estimating advantage function"):
            allret = []
            allb = []
            for path in paths:
                path["returns"] = discount(path["rewards"], gamma)
                b = vf.predict(path)
                b1 = np.append(b,0)
                # b1 = np.append(b, 0 if path["terminated"] else b[-1])
                deltas = path["rewards"] + gamma*b1[1:] - b1[:-1] 
                path["advantage"] = discount(deltas, gamma*lam)
                allb.append(b)
                allret.append(path["returns"])
            baseline_ev = explained_variance_1d(np.concatenate(allb), np.concatenate(allret))
            # baseline_ev = what fraction of variance of returns is explained by the baseline function
            # it'll be <= 1; if it's <= 0, it's giving predictions worse than a constant function.


        with Message("Updating policy"):

            pdist_np = np.concatenate([path["pdists"] for path in paths])
            obs_no = np.concatenate([path["observations"] for path in paths])
            action_na = np.concatenate([path["actions"] for path in paths])
            # Standardize the advantage function to have mean=0 and std=1
            advantage_n = np.concatenate([path["advantage"] for path in paths])
            advantage_n -= advantage_n.mean()
            advantage_n /= (advantage_n.std()+1e-8)

            assert obs_no.shape[0] == pdist_np.shape[0] == action_na.shape[0] == advantage_n.shape[0]

            n_train_paths = int(len(paths)*.75)
            train_sli = slice(0,sum(pathlength(path) for path in paths[:n_train_paths]))
            test_sli = slice(train_sli.stop, None)

            # Training/test split
            poar_train, poar_test = [tuple(arr[sli] for arr in (pdist_np, obs_no, action_na, advantage_n)) for sli in (train_sli, test_sli)]

            obj_names = ["L","KL"]
            obj_before_train = policy.compute_surr_kl(*poar_train)
            obj_before_test = policy.compute_surr_kl(*poar_test)

            def fpen(th):
                thprev = policy.get_parameters_flat()
                policy.set_parameters_flat(th)
                surr, kl = policy.compute_surr_kl(*poar_train) #pylint: disable=W0640
                out = penalty_coeff * kl - surr
                if kl > max_kl or not np.isfinite(out): 
                    out = 1e10
                # testsurr = policy.compute_surr_kl(*poar_test)[0]
                # print "dtheta norm",np.linalg.norm(th - theta),"train lagrangian",out
                # print "testsurr improvement",testsurr - obj_before_test[0]
                policy.set_parameters_flat(thprev)
                return out
            def fgradpen(th):
                thprev = policy.get_parameters_flat()
                policy.set_parameters_flat(th)
                out = - policy.compute_grad_lagrangian(penalty_coeff, *poar_train) #pylint: disable=W0640
                policy.set_parameters_flat(thprev)
                return out                

            theta,_,info = scipy.optimize.fmin_l_bfgs_b(fpen, theta, fprime=fgradpen, maxiter=20)
            del info["grad"]
            print info

            policy.set_parameters_flat(theta)

            obj_after_train = policy.compute_surr_kl(*poar_train)
            obj_after_test = policy.compute_surr_kl(*poar_test)

            delta_train = np.array(obj_after_train) - np.array(obj_before_train)
            delta_test = np.array(obj_after_test) - np.array(obj_before_test)


        with Message("Computing baseline function for next iter"):
            vf.fit(paths)

        with Message("Computing stats"):
            episoderewards = np.array([path["rewards"].sum() for path in paths])
            pathlengths = np.array([pathlength(path) for path in paths])

            entropy = policy.compute_entropy(pdist_np).mean()
            perplexity = np.exp(entropy)

            stats = OrderedDict()
            for (name, dtrain, dtest) in zip(obj_names, delta_train, delta_test):
                stats.update({
                    u"Train_d"+name : dtrain,
                    u"Test_d"+name : dtest,
                    # u"Ratio_"+name : dtest/dtrain
                })
            # stats["Armijo"] = (obj_after_train[0] - obj_before_train[0]) / step.dot(g)
            numeptotal += len(episoderewards)
            stats["NumEpBatch"] = len(episoderewards)
            stats["NumEpTotal"] = numeptotal
            stats["EpRewMean"] = episoderewards.mean()
            stats["EpRewSEM"] = episoderewards.std()/np.sqrt(len(paths))
            stats["EpRewMax"] = episoderewards.max()
            stats["EpLenMean"] = pathlengths.mean()
            stats["EpLenMax"] = pathlengths.max()
            stats["RewPerStep"] = episoderewards.sum()/pathlengths.sum()
            stats["BaselineEV"] = baseline_ev
            stats["Entropy"] = entropy
            stats["Perplexity"] = perplexity
            stats["TimeElapsed"] = time.time() - start_time

        yield stats
