import cgt
from cgt import nn
from param_collection import ParamCollection
from rl import Serializable
from categorical import cat_sample, cat_entropy
from ppo import PPOPolicy

class AtariRAMPolicy(PPOPolicy, Serializable):
    def __init__(self, n_actions):
        Serializable.__init__(self, n_actions)
        cgt.set_precision('double')
        n_in = 128
        o_no = cgt.matrix("o_no",fixed_shape=(None,n_in))
        a_n = cgt.vector("a_n",dtype='i8')
        q_n = cgt.vector("q_n")
        oldpdist_np = cgt.matrix("oldpdists")

        h0 = (o_no - 128.0)/128.0 
        nhid = 64
        h1 = cgt.tanh(nn.Affine(128,nhid,weight_init=nn.IIDGaussian(std=.1))(h0))
        probs_na = nn.softmax(nn.Affine(nhid,n_actions,weight_init=nn.IIDGaussian(std=0.01))(h1))
        logprobs_na = cgt.log(probs_na)
        b = cgt.size(o_no, 0)
        logps_n = logprobs_na[cgt.arange(b), a_n]
        surr = (logps_n*q_n).mean()
        kl = (oldpdist_np * cgt.log(oldpdist_np/probs_na)).sum(axis=1).mean()

        params = nn.get_parameters(surr)
        gradsurr = cgt.grad(surr, params)
        flatgrad = cgt.concatenate([p.flatten() for p in gradsurr])

        lam = cgt.scalar()
        penobj = surr - lam * kl
        self._f_grad_lagrangian = cgt.function([lam, oldpdist_np, o_no, a_n, q_n], 
            cgt.concatenate([p.flatten() for p in cgt.grad(penobj,params)]))
        self.f_pdist = cgt.function([o_no], probs_na)

        self.f_probs = cgt.function([o_no], probs_na)
        self.f_surr_kl = cgt.function([oldpdist_np, o_no, a_n, q_n], [surr, kl])
        self.f_gradlogp = cgt.function([oldpdist_np, o_no, a_n, q_n], flatgrad)

        self.pc = ParamCollection(params)

    def step(self, X):
        pdist_na = self.f_probs(X)
        acts_n = cat_sample(pdist_na)
        return {
            "action" : acts_n,
            "pdist" : pdist_na
        }

    def compute_gradient(self, pdist_np, o_no, a_n, q_n):
        return self.f_gradlogp(pdist_np, o_no, a_n, q_n)

    def compute_surr_kl(self, pdist_np, o_no, a_n, q_n):
        return self.f_surr_kl(pdist_np, o_no, a_n, q_n)

    def compute_grad_lagrangian(self, lam, pdist_np, o_no, a_n, q_n):
        return self._f_grad_lagrangian(lam, pdist_np, o_no, a_n, q_n)

    def compute_entropy(self, pdist_np):
        return cat_entropy(pdist_np)

    def get_parameters_flat(self):
        return self.pc.get_value_flat()

    def set_parameters_flat(self,th):
        return self.pc.set_value_flat(th)




