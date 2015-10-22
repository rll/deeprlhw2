import cgt, numpy as np


class ParamCollection(object):

    """
    A utility class containing a collection of parameters
    which makes it convenient to write optimization code that uses flat vectors
    """

    def __init__(self,params): #pylint: disable=W0622
        """
        params should be a list of cgt nodes that were created by the cgt.shared or nn.parameter functions
        """
        assert all(param.is_data() and param.dtype == cgt.floatX for param in params)
        self._params = params

    @property
    def params(self):
        return self._params

    def get_values(self):
        """
        Returns list of values of parameter arrays
        """
        return [param.op.get_value() for param in self._params]

    def get_shapes(self):
        """
        Shapes of parameter arrays
        """
        return [param.op.get_shape() for param in self._params]

    def get_total_size(self):
        """
        Total number of parameters
        """
        return sum(np.prod(shape) for shape in self.get_shapes())

    def num_vars(self):
        """
        Numbe of parameter arrays
        """
        return len(self._params)

    def set_values(self, parvals):
        """
        Set values of parameter arrays given list of values `parvals`
        """
        assert len(parvals) == len(self._params)
        for (param, newval) in zip(self._params, parvals):
            param.op.set_value(newval)
            assert param.op.get_shape() == newval.shape

    def set_value_flat(self, theta):
        """
        Set parameters using a vector which represents all of the parameters flattened and concatenated
        """
        theta = theta.astype(cgt.floatX)
        arrs = []
        n = 0        
        for shape in self.get_shapes():
            size = np.prod(shape)
            arrs.append(theta[n:n+size].reshape(shape))
            n += size
        assert theta.size == n
        self.set_values(arrs)
    
    def get_value_flat(self):
        """
        Flatten all parameter arrays into one vector and return it as a numpy array
        """
        theta = np.empty(self.get_total_size(),dtype=cgt.floatX)
        n = 0
        for param in self._params:
            s = param.op.get_size()
            theta[n:n+s] = param.op.get_value().flat
            n += s
        assert theta.size == n
        return theta

    def _params_names(self):
        out = []
        for (i,param) in enumerate(self._params):
            name = param.name or _tensordesc(param.typ)
            name = "%s@%i"%(name,i)
            out.append((param,name))
        return out

    def to_h5(self,grp):
        """
        Save parameter arrays to hdf5 group `grp`
        """
        for (param,name) in self._params_names():
            arr = param.op.get_value()
            grp[name] = arr

    def from_h5(self,grp):
        """
        Load parameter arrays from hdf5 group `grp`
        """
        parvals = [grp[name].value for (_,name) in self._params_names()]
        self.set_values(parvals)

def _tensordesc(typ):
    if typ.ndim == 0:
        part0 = "scalar"
    elif typ.ndim == 1:
        part0 = "vector"
    elif typ.ndim == 2:
        part0 = "matrix"
    else:
        part0 = "tensor"+str(typ.ndim)

    return "%s_%s"%(part0, typ.dtype) 

