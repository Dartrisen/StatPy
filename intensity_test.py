__author__ = 'UID_0'
__version__ = '0.0.8'

try:
    import sys
    import time
    import numpy as np
    import itertools as it
    from scipy import arange
    from copy import deepcopy
    from multiprocessing import Pool
    import scipy.integrate as integrate
    from scipy.interpolate import interp1d
    from multiprocessing.dummy import Pool as ThreadPool
except ImportError as e:
    print("couldn't load module {}".format(e))

def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print('%r %2.2f ms' %  (method.__name__, (te - ts)*1000))
        return result
    return timed

class OverTime(Exception):
    def __init__(self):
        super().__init__("Full tau_max is exceed")

class StatPy(object):
    """
    """
    def __init__(self, size = None, mdict = None):
        if (size == None):
            size = 200
        # free surface velocity
        self.vs         = 2.00001
        # time accuracy and max time
        self.tau_max    = 10
        self.dt         = self.tau_max/(size)

        self.eps        = self.dt/10

        self._t         = np.zeros(shape = (size + 1, ))
        self._z         = np.zeros(shape = (size + 1, ))
        self._v         = np.zeros(shape = (size + 1, ))

        self.n          = np.zeros(shape = (size, ))
        self.v          = np.zeros(shape = (size, ))
        self.z          = np.zeros(shape = (size, ))

        if (mdict == None):
            self.mdict ={
                            'path': 'images/' + 'scattering' + '_',
                            'window_title': 'Particle distribution',
                            'xscale': 'linear',
                            'yscale': 'linear',
                            'xlim_min': 0.8,
                            'xlim_max': 4.0,
                            'ylim_min': 0.0,
                            'ylim_max': 0.12,
                            'set_xlabel': r'$\frac{\omega}{2k_0}$',#r'$\frac{\omega}{2k_0}$'
                            'set_ylabel': r'$I(\omega)$'
                        }
        elif (isinstance(mdict, dict)):
            self.mdict = mdict
        else:
            raise TypeError("Wrong type for mdict!")

    def getAttr(self):
        return deepcopy(self.v), deepcopy(self.n)

    def calcTrajectories(self, d, v0, t):
        """
        calculate particle trajectories
        t = [], z = [], v = []
        """
        a       = 0.4*10**(-4)
        tau     = 0
        tau_mid = 0
        #shockwave velocity
        vsh = 1.25*self.vs

        if (v0 < vsh):
            counter = it.count()
            k = next(counter)
            self._t[k] = 0
            self._z[k] = 0
            self._v[k] = v0

            while (tau < self.tau_max):
                k = next(counter)
                tau += self.dt
                self._z[k] = d/(5*a)*np.log(1 + 5.0/d*abs(v0 - self.vs)*tau) + self.vs*tau
                self._v[k] = (v0 - self.vs)/(1 + 5.0/d*(v0 - self.vs)*tau) + self.vs
                self._t[k] = tau
        else:
            counter = it.count()
            k = next(counter)
            #k = 0
            self._t[k] = 0
            self._z[k] = 0
            self._v[k] = v0

            while (tau < self.tau_max):
                tau += self.dt
                if (d*np.log(1 + (1/d)*v0*tau) <= vsh*tau):
                    #print(d*np.log(1 + (1/d)*v0*tau) - vsh*tau)
                    tau_mid = tau
                    #print(tau)
                    break
                else:
                    k = next(counter)
                    self._z[k] = d/a*np.log(1 + 1/d*v0*tau)
                    self._v[k] = v0/(1 + 1/d*v0*tau)
                    self._t[k] = tau

            if (tau >= self.tau_max):
                    #print(tau)
                #raise OverTime()
                idx = np.where(abs(self._t - t) <= self.eps)
                return self._z[idx], self._v[idx]
            else:
                k = next(counter)
                v_mid = v0/(1 + 1/d*v0*tau_mid)
                self._z[k] = vsh*tau_mid
                self._v[k] = v_mid
                self._t[k] = tau_mid

                tau = tau_mid

                if (v_mid < self.vs):
                    while (tau < self.tau_max):
                        k = next(counter)
                        tau += self.dt
                        self._z[k] = (-d/(5*a)*np.log(1 + 5.0/d*(self.vs-v_mid)*(tau-tau_mid))
                                        + self.vs*tau + tau_mid*0.25*self.vs)
                        self._v[k] = ((v_mid - self.vs)/(1 - 5.0/d*(v_mid - self.vs)*(tau-tau_mid))
                                        + self.vs)
                        self._t[k] = tau

                if (v_mid > self.vs):
                    while (tau < self.tau_max):
                        k = next(counter)
                        tau += self.dt
                        self._z[k] = (d/(5*a)*np.log(1+5.0/d*(v_mid - self.vs)*(tau - tau_mid))
                                        + self.vs*tau + tau_mid*0.25*self.vs)
                        self._v[k] = ((v_mid - self.vs)/(1 + 5.0/d*(v_mid - self.vs)*(tau - tau_mid))
                                            + self.vs)
                        self._t[k] = tau
        idx = np.where(abs(self._t - t) <= self.eps)
        return self._z[idx], self._v[idx]
    #@timeit
    def calcDistr(self, d, t, grid, form = 'zero'):
        """
        calculate distribution n(d,u) and u, z main lists for time = t
        """
        counter = it.count()
        for item in grid:
            k = next(counter)
            self.n[k]            = self.distribution(d, item, form)
            self.z[k], self.v[k] = self.calcTrajectories(d, item, t)
    @timeit
    def d_Distr(self, t, grid):
        """
        """
        d_max   = 100
        f       = []
        #f       = np.zeros(shape=(d_max, 1), dtype = type)
        v_min   = np.zeros(shape=(d_max, ))
        v_max   = np.zeros(shape=(d_max, ))

        for d in range(1, d_max):
            print('%r percents completed' % d)
            self.calcDistr(d, t, grid)
            v_min[d-1] = min(self.v)
            v_max[d-1] = max(self.v)
            f.append(interp1d(self.v, self.n, bounds_error = False, fill_value = 0.0))

        items    = np.zeros_like(self.n)
        grid_new = np.linspace(min(v_min), max(v_max), len(grid))

        for item in f:
            items = items + item(grid_new)
        self.n = np.array(items, copy = False)
        self.v = np.array(grid_new, copy = False)
        return

    def tau(self, t, grid):
        """
        """
        def v(z):
            idx = np.where(self.z == z)
            return self.v[idx]

        d_max   = 100
        f       = []
        z_min   = np.zeros(shape=(d_max, ))
        z_max   = np.zeros(shape=(d_max, ))

        for d in range(1, d_max):
            print(d)
            self.calcDistr(d, t, grid)
            z_min[d-1] = min(self.z)
            z_max[d-1] = max(self.z)
            f.append(interp1d(self.z, self.n*self.v, bounds_error = False, fill_value = 0.0))

        items    = np.zeros_like(self.n)
        grid_new = np.linspace(min(z_min), max(z_max), len(grid))

        for item in f:
            items = items + item(grid_new)
        self.n = np.array(items, copy = False)
        self.z = np.array(grid_new, copy = False)
        return deepcopy(self.z), deepcopy(self.n)
    @timeit
    def calcIntensity(self, t):
        grid_w = np.linspace(self.vs/2, 2*self.vs, 200)

        d_max   = 100
        f       = []
        v_min   = np.zeros(shape=(d_max, ))
        v_max   = np.zeros(shape=(d_max, ))

        for d in range(1, d_max):
            print(d)
            self.calcDistr(d, t, grid_w)
            v_min[d-1] = min(self.v)
            v_max[d-1] = max(self.v)
            f.append(interp1d(self.v, np.exp(-2*self.n*self.v)*self.n,
                            bounds_error = False, fill_value = 0.0))

        items    = np.zeros_like(np.exp(-2*self.n*self.v)*self.n)
        grid_new = np.linspace(min(v_min), max(v_max), len(grid_w))

        for item in f:
            items = items + item(grid_new)
        self.n = np.array(items, copy = False)
        self.v = np.array(grid_new, copy = False)
        return

    @staticmethod
    def distribution(d, v, form = None, delta = 0.2):
        """
        exponential approximation of Sorenson data
        """
        if (form == None):
            form = 'zero'
        a = -0.418714 + 0.586249*v -0.0608715*v**2
        A = (np.pi/6)*(integrate.quad(lambda d: d**3*np.exp(-a*d), 1, 100)[0])
        A = (1/delta)*np.exp(-(v-1)/delta)/A
        #A = (3/4*np.pi)*(1/delta)*np.exp(-(v-1)/delta)*np.exp(a)*a**4/(6 + 6*a + 3*a**2 + a**3)

        mdict ={
                    'zero': A*np.exp(-a*d),
                    'quad': A*np.exp(-a*d)*d**2,
                    'cubic': A*np.exp(-a*d)*d**3
                }
        return mdict[form]

    def plotDistr(self, d, t, graphType = None):
        """
        plots a histogram
        """
        import matplotlib
        matplotlib.rcParams['text.usetex'] = True
        import matplotlib.pylab as pl
        import numpy as np

        fig = pl.figure(figsize = (11,11))
        fig.canvas.set_window_title(self.mdict['window_title'])

        plot = fig.add_subplot(1, 1, 1)
        plot.set_xscale(self.mdict['xscale'])
        plot.set_yscale(self.mdict['yscale'])
        #plot.set_xlim(self.mdict['xlim_min'], self.mdict['xlim_max'])
        #plot.set_ylim(self.mdict['ylim_min'], self.mdict['ylim_max'])
        plot.tick_params(axis = 'both', which = 'major', labelsize = 28)
        plot.set_xlabel(r'' + self.mdict['set_xlabel'], fontsize = 28)
        plot.set_ylabel(r'' + self.mdict['set_ylabel'], fontsize = 28)
        plot.set_title(r'$\frac{t}{t_{0}}=$' +' '+ str(t) + ', '
                        + r'$t_{0} = \frac{d_{min}}{av_s}$'
                        + ', ' + r'$d_{min} = 1~\mu m$',fontsize = 28)
        try:
            pl.grid(True)
            if (graphType == None):
                pl.hist(self.v, bins = 50, weights = self.n, normed = False,
                        label = 'd = ' + str(d) + r'$\mu m$')
            else:
                pl.plot(self.v, self.n)
            #pl.legend()
            pl.savefig(self.mdict['path'] + 't=' + str(t) + 'd=' + str(d) + '.png')
        except Exception as e:
            print('{}'.format(e))
        finally:
            #pl.show()
            pl.close()
        return

def makeGrid(v_min, v_max, size = None):
    """
    """
    if (size == None):
        size = 200
    return np.linspace(v_min, v_max, size)

def main():
    t = 0.0
    stat = StatPy()
    grid = makeGrid(1, 4)
    #stat.calcTrajectories(10, 1.001, t)
    #stat.calcDistr(1, t, grid)
    #stat.tau(t, grid)
    stat.d_Distr(t, grid)
    #stat.calcIntensity(t)
    stat.plotDistr('d', t, graphType = None)

if __name__ == '__main__':
    main()
