import numpy as np
import scipy.signal, time

def discount(x, gamma):
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    
    if isinstance(x, float): rep = "%g"%x
    else: rep = unicode(x)
    return u" "*(l - len(rep)) + rep

def fmt_row(widths, row, header=False):
    if isinstance(widths, int): widths = [widths]*len(row)
    out = u" | ".join(fmt_item(x, width) for (x,width) in zip(row,widths))
    if header: out = out + "\n" + "-"*len(out)
    return out

def explained_variance_1d(ypred,y):
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(unicode(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


MESSAGE_DEPTH = 0
class Message(object):
    def __init__(self, msg):
        self.msg = msg
    def __enter__(self):
        global MESSAGE_DEPTH #pylint: disable=W0603
        print colorize('\t'*MESSAGE_DEPTH + '=: ' + self.msg,'magenta')
        self.tstart = time.time()
        MESSAGE_DEPTH += 1
    def __exit__(self, etype, *args):
        global MESSAGE_DEPTH #pylint: disable=W0603
        MESSAGE_DEPTH -= 1
        maybe_exc = "" if etype is None else " (with exception)"
        print colorize('\t'*MESSAGE_DEPTH + "done%s in %.3f seconds"%(maybe_exc, time.time() - self.tstart), 'magenta')


class Timers(object):
    def __init__(self):
        self.key2tc = {}

    def wrap(self,fn,key):
        assert key not in self.key2tc
        self.key2tc[key] = (0,0)
        def timedfn(*args):
            tstart = time.time()
            out = fn(*args)
            (told,cold) = self.key2tc[key]
            dt = time.time() - tstart
            self.key2tc[key] = (told+dt, cold+1)
            return out
        return timedfn

    def stopwatch(self, key):
        if key not in self.key2tc:
            self.key2tc[key]=(0,0)
        class ScopedTimer(object):
            def __enter__(self):
                self.tstart = time.time()
            def __exit__(self1,*_args): #pylint: disable=E0213
                told,cold = self.key2tc[key]
                dt = time.time()-self1.tstart
                self.key2tc[key] = (told + dt, cold + 1)
        return ScopedTimer()

    def disp(self, s="elapsed time"):
        header = "******** %s ********"%s
        print header
        rows = [(key, t, c, t/c) for (key,(t,c)) in self.key2tc.items() if c>0]
        from tabulate import tabulate
        print tabulate(rows, headers=["desc","total","count","per"])
