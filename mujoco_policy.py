import cgt
from cgt import nn
from param_collection import ParamCollection
from rl import Serializable
import numpy as np
from ppo import PPOPolicy

class MujocoPolicy(PPOPolicy, Serializable):
    def __init__(self, obs_dim, ctrl_dim):

        cgt.set_precision('double')
        Serializable.__init__(self, obs_dim, ctrl_dim)

        self.obs_dim = obs_dim
        self.ctrl_dim = ctrl_dim

        o_no = cgt.matrix("o_no",fixed_shape=(None,obs_dim))
        a_na = cgt.matrix("a_na",fixed_shape = (None, ctrl_dim))
        adv_n = cgt.vector("adv_n")
        oldpdist_np = cgt.matrix("oldpdist", fixed_shape=(None, 2*ctrl_dim))
        self.logstd = logstd_1a = nn.parameter(np.zeros((1, self.ctrl_dim)), name="std_1a")
        std_1a = cgt.exp(logstd_1a)

        # Here's where we apply the network
        h0 = o_no
        nhid = 32
        h1 = cgt.tanh(nn.Affine(obs_dim,nhid,weight_init=nn.IIDGaussian(std=0.1))(h0))
        h2 = cgt.tanh(nn.Affine(nhid,nhid,weight_init=nn.IIDGaussian(std=0.1))(h1))
        mean_na = nn.Affine(nhid,ctrl_dim,weight_init=nn.IIDGaussian(std=0.01))(h2)

        b = cgt.size(o_no, 0)
        std_na = cgt.repeat(std_1a, b, axis=0)

        oldmean_na = oldpdist_np[:, 0:self.ctrl_dim]
        oldstd_na = oldpdist_np[:, self.ctrl_dim:2*self.ctrl_dim]

        logp_n = ((-.5) * cgt.square( (a_na - mean_na) / std_na ).sum(axis=1)) - logstd_1a.sum()
        oldlogp_n = ((-.5) * cgt.square( (a_na - oldmean_na) / oldstd_na ).sum(axis=1)) - cgt.log(oldstd_na).sum(axis=1)

        ratio_n = cgt.exp(logp_n - oldlogp_n)

        surr = (ratio_n*adv_n).mean()

        pdists_np = cgt.concatenate([mean_na, std_na], axis=1)
        # kl = cgt.log(sigafter/)

        params = nn.get_parameters(surr)

        oldvar_na = cgt.square(oldstd_na)
        var_na = cgt.square(std_na)
        kl = (cgt.log(std_na / oldstd_na) + (oldvar_na + cgt.square(oldmean_na - mean_na)) / (2 * var_na) - .5).sum(axis=1).mean()


        lam = cgt.scalar()
        penobj = surr - lam * kl
        self._compute_surr_kl = cgt.function([oldpdist_np, o_no, a_na, adv_n], [surr, kl])
        self._compute_grad_lagrangian = cgt.function([lam, oldpdist_np, o_no, a_na, adv_n], 
            cgt.concatenate([p.flatten() for p in cgt.grad(penobj,params)]))
        self.f_pdist = cgt.function([o_no], pdists_np)

        self.f_objs = cgt.function([oldpdist_np, o_no, a_na, adv_n], [surr, kl])

        self.pc = ParamCollection(params)

    def compute_surr_kl(self, *args):
        return self._compute_surr_kl(*args)

    def compute_grad_lagrangian(self, *args):
        return self._compute_grad_lagrangian(*args)

    def get_stdev(self):
        return np.exp(self.logstd.op.get_value().ravel())

    def step(self, X):
        pdist_na = self.f_pdist(X)
        acts_n = meanstd_sample(pdist_na)
        return {
            "action" : acts_n,
            "pdist" : pdist_na
        }

    def compute_entropy(self, pdist_np):
        return meanstd_entropy(pdist_np)

    def pdist_ndim(self):
        return 2*self.ctrl_dim

    def get_parameters_flat(self):
        return self.pc.get_value_flat()

    def set_parameters_flat(self,th):
        return self.pc.set_value_flat(th)



def meanstd_sample(meanstd_np):
    d = meanstd_np.shape[1]//2
    mean = meanstd_np[:,0:d]
    std = meanstd_np[:,d:2*d]
    return mean + std * np.random.randn(*std.shape)

def meanstd_entropy(meanstd_np):
    d = meanstd_np.shape[1]//2
    std_np = meanstd_np[:,d:2*d]
    return np.log(std_np).sum(axis=1)
