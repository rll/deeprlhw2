from rl import animate_rollout, ValueFunction, pathlength
from atari import AtariMDP
import ppo
import numpy as np
from tabulate import tabulate
from prepare_h5_file import prepare_h5_file
import argparse
from atari_ram_policy import AtariRAMPolicy



class AtariRamLinearValueFunction(ValueFunction):
    coeffs = None
    def _features(self, path):
        o = path["observations"].astype('float64')/256.0 - 128.0
        l = pathlength(path)
        al = np.arange(l).reshape(-1,1) / 50.0
        return np.concatenate([o, al, al**2, np.ones((l,1))], axis=1)
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        self.coeffs = np.linalg.lstsq(featmat, returns)[0]
    def predict(self, path):
        return np.zeros(pathlength(path)) if self.coeffs is None else self._features(path).dot(self.coeffs)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--outfile")
    parser.add_argument("--metadata")
    parser.add_argument("--plot",type=int,default=0)
    parser.add_argument("--game",type=str,choices=["pong","breakout","enduro","beam_rider","space_invaders","seaquest","qbert"],default='pong')

    # Parameters
    parser.add_argument("--n_iter",type=int,default=1000)
    parser.add_argument("--gamma",type=float,default=.98)
    parser.add_argument("--lam",type=float,default=1.00)
    parser.add_argument("--timesteps_per_batch",type=int,default=30000)
    parser.add_argument("--penalty_coeff",type=float,default=0.5)
    parser.add_argument("--max_pathlength",type=int,default=10000)
    parser.add_argument("--max_kl",type=float,default=.04)

    args = parser.parse_args()

    np.random.seed(args.seed)

    mdp = AtariMDP('atari_roms/%s.bin'%args.game)
    policy = AtariRAMPolicy(mdp.n_actions)
    vf = AtariRamLinearValueFunction()


    hdf, diagnostics = prepare_h5_file(args, {"policy" : policy, "mdp" : mdp})


    for (iteration,stats) in enumerate(ppo.run_ppo(
            mdp, policy, 
            vf=vf,
            gamma=args.gamma,
            lam=args.lam,
            max_pathlength = args.max_pathlength,
            timesteps_per_batch = args.timesteps_per_batch,
            n_iter = args.n_iter,
            parallel=True,
            max_kl = 0.04,
            penalty_coeff=args.penalty_coeff)):

        print tabulate(stats.items())

        for (statname, statval) in stats.items():
            diagnostics[statname].append(statval)

        if args.plot:
            animate_rollout(mdp,policy,delay=.001,horizon=100)

        grp = hdf.create_group("snapshots/%.4i"%(iteration))
        policy.pc.to_h5(grp)

if __name__ == "__main__":
    main()

