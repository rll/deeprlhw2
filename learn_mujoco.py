from rl import ValueFunction, animate_rollout
import mjcmdp
import ppo
import numpy as np
from tabulate import tabulate
import argparse
from prepare_h5_file import prepare_h5_file
from mujoco_policy import MujocoPolicy

def pathlength(path):
    return path["rewards"].shape[0]

class MujocoLinearValueFunction(ValueFunction):
    coeffs = None
    def _features(self, path):
        o = np.clip(path["observations"], -10,10)
        l = pathlength(path)
        al = np.arange(l).reshape(-1,1)/100.0
        return np.concatenate([o, o**2, al, al**2, al**3, np.ones((l,1))], axis=1)
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        self.coeffs = np.linalg.lstsq(featmat, returns)[0]
    def predict(self, path):
        return np.zeros(pathlength(path)) if self.coeffs is None else self._features(path).dot(self.coeffs)




def main():
    # Command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--outfile")
    parser.add_argument("--metadata")
    parser.add_argument("--plot",type=int,default=0)

    parser.add_argument("--just_sim",action="store_true")

    # Parameters
    parser.add_argument("--n_iter",type=int,default=1000)
    parser.add_argument("--gamma",type=float,default=.99)
    parser.add_argument("--lam",type=float,default=1.0)
    parser.add_argument("--timesteps_per_batch",type=int,default=50000)
    parser.add_argument("--penalty_coeff",type=float,default=0.5)
    parser.add_argument("--max_pathlength",type=int,default=1000)
    args = parser.parse_args()

    # mdp = mjcmdp.CartpoleMDP()
    np.random.seed(args.seed)

    mdp = mjcmdp.HopperMDP()
    (_,(ctrl_dim,)) = mdp.action_spec()
    (_,(obs_dim,)) = mdp.observation_spec()

    policy = MujocoPolicy(obs_dim, ctrl_dim)

    # Saving to HDF5
    hdf, diagnostics = prepare_h5_file(args, {"policy" : policy, "mdp" : mdp})
    vf = MujocoLinearValueFunction()

    for (iteration,stats) in enumerate(ppo.run_ppo(
            mdp, policy, 
            vf=vf,
            gamma=args.gamma,
            lam=args.lam,
            max_pathlength = args.max_pathlength,
            timesteps_per_batch = args.timesteps_per_batch,
            n_iter = args.n_iter,
            parallel=False,
            penalty_coeff=args.penalty_coeff)):
        std_a = policy.get_stdev()
        for (i,s) in enumerate(std_a): stats["std_%i"%i] = s
        print tabulate(stats.items())
        for (statname, statval) in stats.items():
            diagnostics[statname].append(statval)

        if args.plot:
            animate_rollout(mdp,policy,delay=.001,horizon=args.max_pathlength)

        grp = hdf.create_group("snapshots/%.4i"%(iteration))
        policy.pc.to_h5(grp)

if __name__ == "__main__":
    main()

