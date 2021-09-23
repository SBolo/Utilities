import numpy as np
import scipy.optimize as optimize
import pickle
import scipy.stats as scs
from itertools import combinations
from . import tools
if tools.is_notebook(): #the progress bar is different if shell or notebook!
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class Reweight():
    def __init__(self, data, w0=[], optimizer='L-BFGS-B', options={'maxiter': 10000, 'maxfun': 10000000}):
        """
        opt: custom optimizer options
        """
        self.__data = data # dataset class
        self.__nblocks = data.get_nblocks()
        if w0 == [] and self.__nblocks > 0:
            self.__w0 = np.full(self.__nblocks, 1./self.__nblocks) # initial weights
        elif w0 == [] and self.__nblocks == 0:
            self.__w0 = np.full(self.__nblocks, 0.)
        else:
            self.__w0 = w0

        self.__optimizer = optimizer
        self.__options = options
        
        self.__rates = data.get_rates()
        self.__errors = data.get_errors()
        self.__phi = []
        self.results = {}

        #compute initial reduced chi2 of the dataset
        self.initial_chi2 = {}
        for r in range(data.get_nrates()):
            self.initial_chi2[f'R{r+1}'] = tools.chi2r(self.__rates['experiment'][r], self.__rates['averaged'][r], \
                                   self.__errors['experiment'][r], self.__errors['simulated'][r])
        self.initial_chi2[f'overall'] = tools.chi2r(self.__rates['experiment'], self.__rates['averaged'], \
                                                self.__errors['experiment'], self.__errors['simulated'])
    # ------------------------------------------------------------------------------------------------------------------

    def describe(self):
        print(f"# Reweighting class: \n#  - {self.__optimizer} optimizer\n#  - {self.initial_chi2[f'overall']:.2f}"
              f" initial reduced chi2")

    def phi_eff(self, w):
        """
        Computes the fraction of effective parameters.

        Parameters
        ----------
        w : numpy.ndarray
            array of weights to be optimized

        Returns
        -------
        phi : numpy.ndarray
            array of effective fraction of frames
        """

        idxs = np.where(w > 1e-50)
        srel = np.sum(w[idxs] * np.log(w[idxs] / self.__w0[idxs]))
        phi = np.exp(-srel)
        return phi
    # ------------------------------------------------------------------------------------------------------------------

    def get_prior(self):
        return np.copy(self.__w0)
    # ------------------------------------------------------------------------------------------------------------------

    def reset_prior(self, w0):
        self.__w0 = w0
    # ------------------------------------------------------------------------------------------------------------------

    def get_theta_idx(self, phi_eff):
        """
        Get the index of theta closeset to a given value of phi effective.

        Parameters
        ----------
        phi_eff : float
                  value of phi effective

        Returns
        -------
        idx : int
              index of the corresponding value of theta
        """
        if self.__phi != []:
            idx = (np.abs(np.array(self.__phi) - phi_eff)).argmin()
        else:
            raise ValueError('phi_eff is empty.')
        return idx
    # ------------------------------------------------------------------------------------------------------------------

    def __chi2r(self, w, r):
        rrw = np.dot(self.__rates['simulated'], w[:, np.newaxis])[:, :, 0]  # removes last, useless axis
        if r == -1:
            rex = self.__rates['experiment']
            eex = self.__errors['experiment']
            emd = self.__errors['simulated']
        else:
            rex = self.__rates['experiment'][r]
            eex = self.__errors['experiment'][r]
            emd = self.__errors['simulated'][r]
            rrw = rrw[r]
        chi2 = tools.chi2r(rex, rrw, eex, emd)  # compute chi2
        return chi2
    # ------------------------------------------------------------------------------------------------------------------

    def __penalty(self, w, r, theta):
        """
        Computes the value of the penalty function for a single rate.

        Parameters
        ----------
        w : numpy.ndarray
            array of weights to be optimized
        r : int
            rate index.
        theta : int
            fudge parameter.

        Returns
        -------
        r : float
            value of the penalty function given the set of weights
        """
        w /= np.sum(w)  # normalize weights
        idxs = np.where(w > 1e-50)  # isolate non-zero weights
        rrw = np.dot(self.__rates['simulated'], w[:, np.newaxis])[:, :, 0] # removes last, useless axis
        if r == -1:
            rex = self.__rates['experiment']
            eex = self.__errors['experiment']
            emd = self.__errors['simulated']
        else:
            rex = self.__rates['experiment'][r]
            eex = self.__errors['experiment'][r]
            emd = self.__errors['simulated'][r]
            rrw = rrw[r]
        chi2 = tools.chi2(rex, rrw, eex, emd)
        srel = np.sum(w[idxs] * np.log(w[idxs] / self.__w0[idxs])) # compute relative entropy
        p = 0.5 * chi2 + theta * srel  # sum all the contributions
        return p
    # ------------------------------------------------------------------------------------------------------------------

    def reweight(self, r, thetas, out_name=""):
        """
        Reweights the data with respect to a single rate r.
        It saves the results in a .pkl file.

        Parameters
        ----------
        r : int
            rate index.
        thetas : list
            range of hyperparameter to use in the minimization
        """
        self.results = dict.fromkeys(thetas, 0)
        bounds = [(0, 1)] * len(self.__w0)  # weights have to be bound between 0 and 1
        flags = [] #stores error messages of minimizations that didn't converge

        w0 = self.__w0 #initial guess (not prior!) is reset at every value of theta to accelerate convergence
        print("# REWEIGHTING")
        for t in tqdm(thetas): # run minimization for all values of theta
            rs = optimize.minimize(self.__penalty, w0, args=(r, t), bounds=bounds, jac=False, options=self.__options,
                                   method=self.__optimizer)
            w0 = rs.x / np.sum(rs.x)
            self.results[t] = w0
            if not rs.success:  # some minimizations had problems!
                flags.append([t, rs.message])

        if flags == []:
            print("# All minimizations terminated successfully!")
        else:
            print("# Some minimizations terminated unsuccessfully:")
            for flag in flags:
                print(f"#  - At theta={flag[0]}: {flag[1].decode('utf-8')}")
            print("# These issues might be related to your choice of the optimizer and its options. Give it a try with "
                  "the default settings, change the values of theta or report us the problem!")

        # save results
        if out_name != "":
            tools.save_pickle(self.results, out_name)
            print(f"# Results have been saved as '{out_name}'")
    # ------------------------------------------------------------------------------------------------------------------

    def phi_chi2r(self, r):
        """
        Computes the phi_eff vs reduced chi squared curve for a provided rate.

        Parameters
        ----------
        r : int
            rate index.
        """

        chi2r = []
        phi_eff = []
        for k in self.results.keys():
            w = self.results[k]
            phi_eff.append(self.phi_eff(w))
            x2 = self.__chi2r(w, r)
            chi2r.append(x2)

        return phi_eff, chi2r
    # ------------------------------------------------------------------------------------------------------------------

    def get_theta_idx(self, phi_eff):
        """
        Get the index of theta closeset to a given value of phi effective.

        Input
        -----
        phi_eff : float
                  value of phi effective

        Returns
        -------
        idx : int
              index of the corresponding value of theta
        """
        phi, _ = self.phi_chi2r(self, -1)
        idx = (np.abs(np.array(phi) - phi_eff)).argmin()
        return idx
    # ------------------------------------------------------------------------------------------------------------------

    def set_optimizer_options(self, options):
        """"""
        self.__options = options
    # ------------------------------------------------------------------------------------------------------------------

    def set_optimizer(self, optimizer):
        """"""
        self.__optimizer = optimizer
    # ------------------------------------------------------------------------------------------------------------------

    def load_results(self, to_load):
        """"""
        self.results = self.__data.load_pickle(to_load)
    # ------------------------------------------------------------------------------------------------------------------

    def save_results(self, out_name):
        """"""
        self.__data.save_pickle(self.results, out_name)
