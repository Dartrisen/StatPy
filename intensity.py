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
        self.dt         = self.tau_max/size

        self.eps        = self.dt/10

        self.tlist      = []
        self.zlist      = []
        self.vlist      = []

        self.n          = np.zeros(shape=(size, ))
        self.v          = np.zeros(shape=(size, ))
        self.z          = np.zeros(shape=(size, ))

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

    def clear(self):

        self.zlist.clear()
        self.tlist.clear()
        self.vlist.clear()

    def getAttr(self):

        return deepcopy(self.v), deepcopy(self.n)
    @timeit
    def calcTrajectories(self, d, v, t):
        """
        calculate particle trajectories
        t = [], z = [], v = []
        """
        a       = 0.4*10**(-4)
        tau     = 0
        tau_mid = 0

        self.clear()

        if (v < 1.25*self.vs):
            self.tlist.append(0)
            self.zlist.append(0)
            self.vlist.append(v)

            while (tau < self.tau_max):
                tau += self.dt
                self.zlist.append(d/(5*a)*np.log(1 + 5.0/d*abs(v - self.vs)*tau) + self.vs*tau)
                self.vlist.append((v - self.vs)/(1 + 5.0/d*(v - self.vs)*tau) + self.vs)
                self.tlist.append(tau)
        else:
            self.tlist.clear()
            self.zlist.clear()
            self.vlist.clear()
            self.tlist.append(0)
            self.zlist.append(0)
            self.vlist.append(v)

            while (tau < self.tau_max):
                tau += self.dt
                if (d*np.log(1 + (1/d)*v*tau) <= 1.25*self.vs*tau):
                    tau_mid = tau
                    break
                else:
                    self.zlist.append(d/a*np.log(1 + 1/d*v*tau))
                    self.vlist.append(v/(1 + 1/d*v*tau))
                    self.tlist.append(tau)

            v_mid = v/(1 + 1/d*v*tau_mid)
            self.zlist.append(1.25*self.vs*tau_mid)
            self.vlist.append(v_mid)
            self.tlist.append(tau_mid)

            tau = tau_mid

            if (v_mid < self.vs):
                while (tau < self.tau_max):
                    tau += self.dt
                    self.zlist.append(-d/(5*a)*np.log(1 + 5.0/d*(self.vs-v_mid)*(tau-tau_mid)) + self.vs*tau
                                    + tau_mid*0.25*self.vs)
                    self.vlist.append((v_mid - self.vs)/(1 - 5.0/d*(v_mid - self.vs)*(tau-tau_mid)) + self.vs)
                    self.tlist.append(tau)

            if (v_mid > self.vs):
                while (tau < self.tau_max):
                    tau += self.dt
                    self.zlist.append(d/(5*a)*np.log(1+5.0/d*(v_mid - self.vs)*(tau - tau_mid)) + self.vs*tau
                                    + tau_mid*0.25*self.vs)
                    self.vlist.append((v_mid - self.vs)/(1 + 5.0/d*(v_mid - self.vs)*(tau - tau_mid)) + self.vs)
                    self.tlist.append(tau)
        print(len(self.tlist))
        for item in self.tlist:
            if (abs(t - item) <= self.eps):
                idx = self.tlist.index(item)
        return self.zlist[idx], self.vlist[idx]

    def calcDistr(self, d, t, grid, form = 'quad'):
        """
        calculate distribution n(d,u) and u, z main lists for time = t
        """
        counter = it.count()
        for item in grid:
            count = next(counter)
            self.n[count] = self.distribution(d, item, form)
            self.z[count], self.v[count] = self.calcTrajectories(d, item, t)

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
        plot.set_xlim(self.mdict['xlim_min'], self.mdict['xlim_max'])
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
    t = [0.0, 0.1]
    stat = StatPy()
    grid = makeGrid(1, 4)
    #stat.calcTrajectories(100, 4.0, t)
    #stat.calcDistr(40, 0, grid)
    #stat.tau(0.1, grid)
    #stat.d_Distr(0.0, grid)
    for item in t:
        stat.calcIntensity(item)
        stat.plotDistr('d', item, graphType = 'plot')

if __name__ == '__main__':
    main()
