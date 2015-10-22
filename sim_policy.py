#!/usr/bin/env python
import argparse
from rl import animate_rollout
import cPickle, h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf")
    parser.add_argument("--max_pathlength",type=int,default=1000)
    args = parser.parse_args()

    hdf = h5py.File(args.hdf,'r')

    policy = cPickle.loads(hdf['policy_pickle'].value)
    mdp = cPickle.loads(hdf['mdp_pickle'].value)
    policy.pc.from_h5(hdf['snapshots'].values()[-1])

    while True:
        animate_rollout(mdp,policy,delay=.001,horizon=args.max_pathlength)
        raw_input("press enter to continue (exit with ctrl-C)")

if __name__ == "__main__":
    main()