import h5py, atexit
from collections import defaultdict
import os.path as osp, cPickle

def prepare_h5_file(args, objs_to_pickle):
    outfile_default = "/tmp/a.h5"
    fname = args.outfile or outfile_default
    if osp.exists(fname) and fname != outfile_default:
        raw_input("output file %s already exists. press enter to continue. (exit with ctrl-C)"%fname)
    hdf = h5py.File(fname,"w")
    hdf.create_group('params')
    for (param,val) in args.__dict__.items():
        try: hdf['params'][param] = val
        except (ValueError,TypeError): 
            print "not storing parameter",param
    diagnostics = defaultdict(list)
    print "Saving results to %s"%fname
    def save():
        hdf.create_group("diagnostics")
        for (diagname, val) in diagnostics.items():
            hdf["diagnostics"][diagname] = val        
    atexit.register(save)

    for (key,val) in objs_to_pickle.items():
        hdf[key + "_pickle"] = cPickle.dumps(val)

    return hdf, diagnostics