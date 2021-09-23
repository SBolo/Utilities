from itertools import combinations
import numpy as np
import scipy.stats as scs
import mdtraj as md
from . import tools

class Data:
    def __init__(self):
        """Class to handle ABSURDer data structures"""
        self.__rates= {'experiment':np.array([]), 'simulated':np.array([]), 'averaged':np.array([])}
        self.__errors = {'experiment':np.array([]), 'simulated':np.array([])}
        self.__specdens = {'experimental':np.array([]), 'simulated':np.array([]), 'synthetic':np.array([])}
        self.__rotamers = {'experimental': np.array([]), 'simulated': np.array([]), 'aminos': np.array([])}
        self.__methyls = [] # list of methyl names
        self.__nrates = 0 # number of rates
        self.__nmethyls = 0 # number of methyls
        self.__nblocks = 0 # number of blocks
        self.__ignore = [] # save the indices of ignored methyls for default behaviour in methyl name list loading
        self.__description = False # flag for class description
        self.__names_are_consistent = False # flag to check if length of methyl name list is consistent
        self.close_methyls_indices = {} # indices of close methyls to every other residue in the protein
    # ------------------------------------------------------------------------------------------------------------------

    # LOADING AND SAVING FUNCTIONS
    #-----------------------------
    def __load_rates(self, rate):
        """
        Loads the provided rates depending on their types.
        Raises a ValueError if an invalid type is recognized.

        Input
        -----
        rate : str or numpy.ndarray
            path to a numpy array, path to a pickle or numpy array

        Returns
        -------
        rate : numpy.ndarray
             rate matrix
        """
        if type(rate) == str and rate[-3:] == 'npy':
            r = np.load(rate)
            return r
        elif type(rate) == str and rate[-3:] == 'pkl':
            r = tools.load_pickle(rate)
            #pin = open(rate, "rb")
            #r = pickle.load(pin)
            return r
        elif type(rate) == np.ndarray:
            return rate
        else:
            raise ValueError(f"# Provided rates are '{type(rate)}' but they need to be either numpy.ndarray, "
                             f"a pickle or a string.")
    # ------------------------------------------------------------------------------------------------------------------

    def __ignore_methyls(self, ignore):
        """
        Removes from the dataset a list of specified methyls that need to be ignored.

        Parameters
        ----------
        idx : list
            indices of methyls to be removed
        """
        rex = np.empty((self.__rates['experiment'].shape[0], self.__rates['experiment'].shape[1] - len(ignore)))
        rmd = np.empty((self.__rates['simulated'].shape[0], self.__rates['simulated'].shape[1] - len(ignore), \
                        self.__rates['simulated'].shape[2]))
        eex = np.empty((self.__errors['experiment'].shape[0], self.__errors['experiment'].shape[1] - len(ignore)))

        k = 0
        for j in range(self.__nmethyls):
            if j not in ignore:
                rex[:, k] = self.__rates['experiment'][:, j]
                eex[:, k] = self.__errors['experiment'][:, j]
                rmd[:, k, :] = self.__rates['simulated'][:, j, :]
                k += 1
        self.__rates['experiment'] = rex
        self.__errors['experiment'] = eex
        self.__rates['simulated'] = rmd
    # ------------------------------------------------------------------------------------------------------------------

    def load_data(self, exp_rates, sim_rates, exp_errors=[], ignore=[], verbose=True):
        """
        Loads all the rate datasets required for the ABSURDer calculations

        Input
        -----
        exp_rates : str or numpy.ndarray
            path to the file with the experimental rate matrix or the experimental rate matrix array
        sim_rates : str or numpy.ndarray
            path to the file with the simulated rate matrix or the simulated rate matrix array
        exp_errors : str or numpy.ndarray
            path to the file with the experimental error matrix or the experimental error matrix array. If no errors
            are provided, based on the dimension of exp_rates and sim_rates the function will create a toy model
        ignore : list
            list of methyl indices to ignore (e.g. [3,6,7])
        verbose : bool
            if True, prints extra messages about non-default behaviours
        """
        self.__rates['experiment'] = self.__load_rates(exp_rates)
        self.__rates['simulated'] = self.__load_rates(sim_rates)

        # basic info about the dataset
        self.__nmethyls = self.__rates['experiment'].shape[1]
        self.__nblocks = self.__rates['simulated'].shape[2]
        self.__nrates = self.__rates['experiment'].shape[0]
        self.__ignore = ignore

        # check if experimental errors have been provided, if not build a toy model if the experimental rates matrix
        # is provided in the correct shape, otherwise raise an error
        if len(exp_errors) != 0:
            self.__errors['experiment'] = self.__load_rates(exp_errors)
        else:
            if len(np.shape(self.__rates['experiment'])) == 2:
                if verbose:
                    print(f"# NOTE: no experimental errors provided and {self.__rates['experiment'].shape} dimensional "
                          f"experimental input. Interpreting the experimental data as coming from a single "
                          f"methyl group")
                self.__rates['experiment'] = self.__rates['experiment'][:, np.newaxis, :]
                self.__errors['experiment'] = np.std(self.__rates['experiment'], axis=-1) / np.sqrt(self.__nblocks)
                self.__rates['experiment'] = np.mean(self.__rates['experiment'], axis=-1)
            elif len(np.shape(self.__rates['experiment'])) == 1 or len(np.shape(self.__rates['experiment'])) > 3:
                raise ValueError(f"No experimental errors provided. The expected dimension of experimental rates has "
                                 f"is be nrates x nmethyls x blocks but the current shape is "
                                 f"{self.__rates['experiment'].shape})")
            else:
                tmp = np.copy(self.__rates['experiment'])
                self.__rates['experiment'] = np.mean(tmp, axis=-1)
                self.__errors['experiment'] = np.std(tmp, axis=2)/np.sqrt(tmp.shape[2])
                if verbose:
                    print("# NOTE: synthetic experimental dataset created")

        #check dimension of the experimental rates matrix
        if len(np.shape(self.__rates['experiment'])) == 1:
            if verbose:
                print("# NOTE: interpreting the experimental data as coming from a single methyl group")
            # add extra axis to make sure all calculations will be fine
            self.__rates['experiment'] = self.__rates['experiment'][:, np.newaxis]
            self.__errors['experiment'] = self.__errors['experiment'][:, np.newaxis]
        elif len(np.shape(self.__rates['experiment'])) > 2:
            raise ValueError(f"Dimension of experimental rates has to be nrates x nmethyls "
                             f"(current shape is {self.__rates['experiment'].shape})")

        # check dimension of the simulated rate matrix
        if len(np.shape(self.__rates['simulated'])) == 2:
            if verbose:
                print("# NOTE: interpreting the simulated data as coming from a single methyl group")
            # add extra axis to make sure all calculations will be fine
            self.__rates['simulated'] = self.__rates['simulated'][:, np.newaxis, :]
        elif len(np.shape(self.__rates['simulated'])) > 3:
            raise ValueError(f"Dimension of experimental rates has to be nrates x nmethyls x nblocks "
                             f"(current shape is {self.__rates['simulated'].shape})")

        # check consistency between experimental and simulated rates and errors
        if self.__rates['experiment'].shape[:2] != self.__rates['simulated'].shape[:2] or \
            self.__rates['experiment'].shape[:2] != self.__errors['experiment'].shape[:2]:
            raise ValueError(f"The number of rates and methyls must be identical in the experimental rate matrix "
                             f"{self.__rates['experiment'].shape[:2]}, in the experimental error matrix "
                             f"{self.__errors['experiment'].shape[:2]} and in the simulated rate matrix "
                             f"{self.__rates['simulated'].shape[:2]}")

        # remove ignored methyls
        if ignore:
            self.__ignore_methyls(ignore)
            self.__nmethyls = self.__rates['experiment'].shape[1]
            self.__nblocks = self.__rates['simulated'].shape[2]
            self.__nrates = self.__rates['experiment'].shape[0]

        # compute average over blocks
        self.__rates['averaged'] = np.mean(self.__rates['simulated'], axis=-1)
        self.__errors['simulated'] = np.std(self.__rates['simulated'], axis=-1) / np.sqrt(self.__nblocks)
        self.__description = True
    # ------------------------------------------------------------------------------------------------------------------

    def load_and_sort(self):
        """"""
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def save_dataset(self, filename):
        """"""
        tools.save_pickle(self, filename)
    # ------------------------------------------------------------------------------------------------------------------

    def overwrite_rates(self, rates_dictionary):
        """Overwrite existing rate dictionary"""
        assert rates_dictionary.keys() == self.__rates.keys(), "The dictionary keys must match."
        self.__rates = rates_dictionary
    # ------------------------------------------------------------------------------------------------------------------

    def overwrite_errors(self, errors_dictionary):
        """Overwrite existing errors dictionary"""
        assert errors_dictionary.keys() == self.__errors.keys(), "The dictionary keys must match."
        self.__errors = errors_dictionary
    # ------------------------------------------------------------------------------------------------------------------

    def load_methyls_list(self, methyls_list, ignore=True):
        """
        Loads a list of methyl names. The list is assumed to be in the same order as the experimental and simulated
        rates. If the length of the name list is different from the one of the simulated and experimental rates,
        the list is loaded anyway but it is ignored in the class

        Input
        -----
        methyls_list : str
            path to the methyl name list files or methyl name list
        ignore : bool or list
            methyls indices to ignore. If True, the list of ignored methyls is the same used when loading the rates. If
            False, no methyls will be ignored. Otherwise a list should be provided with the indices to be ignored.
        """
        if ignore:
            ignore = self.__ignore # default behaviour: ignore the same methyls
        elif not ignore:
            ignore = []

        if type(methyls_list) == str:
            #pin = open(methyls_list, "rb")
            self.__methyls = tools.load_pickle(methyls_list) #pickle.load(pin)
        elif type(methyls_list) == list:
            self.__methyls = methyls_list
        else:
            raise ValueError("Methyls list must be either a path to a file or a list.")
        if ignore:
            mm = [m for i,m in enumerate(self.__methyls) if i not in ignore]
            self.__methyls = mm
        if len(self.__methyls) == self.__nmethyls:
            self.__names_are_consistent = True
        else:
            print("# WARNING: the length of the provided list of methyl names is inconsistent with the number"
                  "of methyls present in the rate matrix. The name list will be ignored.")
    # ------------------------------------------------------------------------------------------------------------------

    # MISCELLANEOUS
    #--------------
    def describe(self):
        """Prints a short summary of the class status"""
        if self.__description:
            print(f"# Data class with: \n#  - {self.__nmethyls} methyls\n#  - {self.__nrates} rates\n#  - {self.__nblocks} "
                  f"simulated blocks")
        else:
            print("# Empty Data class")
    # ------------------------------------------------------------------------------------------------------------------

    def methyl_index(self, methyl):
        """Checks if a methyl exists in the methyl name lists and returns the corresponding index

        Input
        -----
        methyl : str
            methyl name

        Returns
        -------
        self.__methyls.index(methyl) : int
            index of the corresponding methyl in the methyl name list
        """
        if methyl in self.__methyls:
            return self.__methyls.index(methyl)
        else:
            raise ValueError(f'Methyl group {methyl} is not in the provided methyl list.')
    # ------------------------------------------------------------------------------------------------------------------

    # GET FUNCTIONS
    #--------------
    def get_rates(self):
        """Return the rates dictionary"""
        return self.__rates
    # ------------------------------------------------------------------------------------------------------------------

    def get_errors(self):
        """Return the error dictionary"""
        return self.__errors
    # ------------------------------------------------------------------------------------------------------------------

    def get_nblocks(self):
        """Return the number of rate blocks"""
        return self.__nblocks
    # ------------------------------------------------------------------------------------------------------------------

    def get_nmethyls(self):
        """Return the number of methyl groups"""
        return self.__nmethyls
    # ------------------------------------------------------------------------------------------------------------------

    def get_nrates(self):
        """Return the number of relaxation rates"""
        return self.__nrates
    # ------------------------------------------------------------------------------------------------------------------

    def get_names_are_consistent(self):
        """Returns a flag for the methyl list name sanity check"""
        return self.__names_are_consistent
    # ------------------------------------------------------------------------------------------------------------------

    def get_methyls(self):
        """Return list of methyls"""
        return self.__methyls
    # ------------------------------------------------------------------------------------------------------------------

    def get_specdens(self):
        return self.__specdens
    # ------------------------------------------------------------------------------------------------------------------

    # CREATING SUBSETS OF DATA
    #-------------------------
    def sample_random_subset(self, n):
        """
        Generate a random subset of the provided methyl rates. Useful for testing or controls.

        Input:
        -----
        n : int
           number of methyls to keep

        Returns:
        -------
        sub : aBSURDer.data.Data
            a data class with the subsampled methyls
        """
        methyl_indices = np.random.choice(np.arange(0, self.__nmethyls, 1), n, replace=True)
        exp = np.copy(self.__rates['experiment'])[:, methyl_indices]
        err = np.copy(self.__errors['experiment'])[:, methyl_indices]
        rmd = np.copy(self.__rates['simulated'])[:, methyl_indices, :]
        if self.__methyls and self.__names_are_consistent:
            methyls = [self.__methyls[i] for i in methyl_indices]

        d = Data()
        d.load_data(np.array(exp), np.array(rmd), np.array(err), verbose=False)
        if self.__methyls and self.__names_are_consistent:
            d.load_methyls_list(methyls)
        return d
    # ------------------------------------------------------------------------------------------------------------------

    def extract_subset(self, methyl_indices=[], methyl_names=[]):
        """
        Extract a subset of methyls from the dataset and create a new data class. Either indices or methyl names can
        be used. Methyl names will raise an error if no methyl name list has been loaded before. If both lists are
        provided, methyl_names will override methyl_indices.

        Input
        -----
        methyl_indices : list
            list of methyl indices to keep
        methyl_names : list
            list of methyl names to keep

        Returns
        -------
        d : aBSURDer.data.Data
            a data class with the subsampled methyls
        """
        if not methyl_indices and not methyl_names:
            raise ValueError('At least one between methyl_indices and methyl_names should be a non empty list.')
        if methyl_names:
            assert self.__methyls, "No list of methyls was loded. Provide a list of methyl indices with methyl_indices" \
                                   "or load a methyl list with load_methyls_list()."
            methyl_indices = [self.methyl_index(m) for m in methyl_names]
            print(methyl_indices)

        exp = np.copy(self.__rates['experiment'])[:, methyl_indices]
        err = np.copy(self.__errors['experiment'])[:, methyl_indices]
        rmd = np.copy(self.__rates['simulated'])[:, methyl_indices, :]

        d = Data()
        d.load_data(np.array(exp), np.array(rmd), np.array(err), verbose=True)
        d.load_data(np.array(exp), np.array(rmd), np.array(err), verbose=True)
        if methyl_names:
            d.load_methyls_list(methyl_names)
            d.__names_are_consistent=True
        elif self.__methyls and self.__names_are_consistent:
            methyls = [self.__methyls[i] for i in methyl_indices]
            d.load_methyls_list(methyls)
            d.__names_are_consistent = True
        return d
    # ------------------------------------------------------------------------------------------------------------------

    def close_methyls(self, pdb, cutoff=0.5, empty_if=0):
        """
        Starting from an existing dataset, create a new dataset based on a methyl and its structurally close
        methyls for which experimental data are available.

        Input
        -----
        pdb : str
            path to protein topology in pdb
        cutoff : float
            distance cutoff within which to look for methyls
        empty_if : int
            minimum number of close methyls required to return a non-empty list
        """
        assert len(self.__methyls) != 0, "No methyl name list has been loaded. Run load_methyls_list() first."
        # create list of available methyl-bearing residues
        midx, mnl = [], []
        for m in self.__methyls:
            m = m.split('-')
            num = int(m[0][3:]) - 1
            res = m[0]
            if num not in midx:
                midx.append(num)
            if res not in mnl:
                mnl.append(res)
        midx = np.array(midx)

        # load the protein pdb and create a distance matrix
        pdb = md.load(pdb)
        distances, residue_pairs = md.compute_contacts(pdb, periodic=False)
        mat = md.geometry.squareform(distances, residue_pairs)[0]

        self.close_methyls_indices = {}
        for i in range(pdb.n_residues):
            # select only relevant residues and sort them by distance, then pick the ones within the cutoff
            mat_methyl = mat[i, :][midx]
            sort_idxs = np.argsort(mat_methyl)
            sortmat = mat_methyl[sort_idxs]
            L = len(sortmat[sortmat < cutoff])
            if L <= empty_if:
                self.close_methyls_indices[i] = []
            else:
                to_use = sort_idxs[:L]
                res_indices = [mnl[i] for i in to_use]
                res_indices = [int(m.split('-')[0][3:]) for m in res_indices]

                methyl_indices = []
                for idx in res_indices:
                    for m in self.__methyls:
                        n = m.split('-')
                        num = int(n[0][3:])
                        if idx == num:
                            methyl_indices.append(self.__methyls.index(m))
                self.close_methyls_indices[i] = list(np.unique(methyl_indices))
        return self.close_methyls_indices
    # ------------------------------------------------------------------------------------------------------------------

    def quantile_filter(self, q, bins=100):
        """
        Keep only a subset of methyls which experimental rates fall within a given quantile of the distribution of
        the simulated rates.

        Input
        -----
        q : float
            value of the quantile
        bins : int
            number of bins to create the rate histogram

        Returns
        -------
        methyls : list
            list of methyls
        """
        methyls = []
        for idx in range(self.__nmethyls):
            keep = []
            for r in range(self.__nrates):
                h, _ = np.histogram(self.__rates['simulated'][r][idx], bins=bins, density=True)
                x = np.linspace(self.__rates['simulated'][r, idx, :].min(), self.__rates['simulated'][r, idx, :].max(),
                                num=bins)
                quant = tools.arg_quantile(h, q)
                keep.append(self.__rates['experiment'][r][idx] < x[quant])
            if np.sum(keep) == 3:
                methyls.append(idx)

        return methyls
    # ------------------------------------------------------------------------------------------------------------------

    def filter_by_keyword(self, key1, key2=""):
        """"""
        assert self.__methyls, "No methyl names list has been loaded. Run load_methyls_list() first."
        methyls = []
        for m in self.__methyls:
            if key1 in m and key2 in m:
                methyls.append(m)
        return methyls
    # ------------------------------------------------------------------------------------------------------------------

    def load_Jsynth(self, jsy, ignore=[]):
        """
        Loads the experimental power spectral density from a pickle file. Useful only in the case of synthetically
        generated data for which an experimental J(w) is available.

        Input
        -----
        jex : str
            path to the pickle file with the power spectral density
        """
        self.__specdens['synthetic'] = tools.load_pickle(jsy).T

        # use
        if not ignore and self.__ignore:
            ignore = self.__ignore

        if ignore:
            jsy = np.empty((self.jsy.shape[0], self.jsy.shape[1] - len(ignore), self.jsy.shape[2]))
            k = 0
            for j in range(self.jex.shape[1]):
                if j not in self.idx:
                    jex[:, k, :] = self.jex[:, j, :]
                    jmd[:, k, :] = self.jmd[:, j, :]
                    jws[:, k, :] = self.jws[:, j, :]
                    k += 1

    # ------------------------------------------------------------------------------------------------------------------

    def load_Jexp(self, jex, ignore=[]):
        """
        Loads the power spectral density in J(0), J(wD) and J(2wD) from a pickle file.

        Input
        -----
        jwd : str
            path to the pickle file with the power spectral density
        """
        self.__specdens['experimental'] = tools.load_pickle(jex).T
    # ------------------------------------------------------------------------------------------------------------------

    def load_Jsim(self, jsim, ignore=[]):
        """
        Loads the power spectral density in J(0), J(wD) and J(2wD) from a pickle file.

        Input
        -----
        jwd : str
            path to the pickle file with the power spectral density
        """
        self.__specdens['simulated'] = tools.load_pickle(jsim).T
    # ------------------------------------------------------------------------------------------------------------------

    def load_specdens(self, jex, jws, jmd):

        """
        Loads into the class the files needed to plot the spectral densities

        Parameters
        ----------
        jex : str
            path to the pickle with the experimental spectral density functions.
        jws : int
            path to the pickle with the values of J(0), J(w) and J(2w).
        jmd : str
            path to the pickle with the simulated spectral density functions.
        """

        pin = open(jex, "rb")
        self.jex = pickle.load(pin)
        self.jex = self.jex.T  # remember to remove this with new data!!

        pin = open(jws, "rb")
        self.jws = pickle.load(pin)
        self.jws = self.jws.T  # remember to remove this with new data!!

        pin = open(jmd, "rb")
        self.jmd = pickle.load(pin)
        self.jmd = self.jmd.T  # remember to remove this with new data!!

        self.ignore_specdens()  # ignore a set of methyl groups
        self.specdens_load = True

    # -----------------------------------------------------------------------------------------------------------------

    def filter_dataset(self):
        pass
        # def filter_for_nmr(data, all_methyls, nmr_methyls):
        #     idxs = []
        #     for m in nmr_methyls:
        #         if m[0] == 'I' and m[-1] == 'D':
        #             ile = str(m.strip().split('-')[0] + '-CD1HD1')
        #             if ile in all_methyls:
        #                 idxs.append(all_methyls.index(ile))
        #         else:
        #             if m in all_methyls:
        #                 idxs.append(all_methyls.index(m))
        #     filtered = []
        #     for r in range(4):
        #         filtered_tmp = []
        #         for i in idxs:
        #             filtered_tmp.append(data[r][i])
        #         filtered.append(filtered_tmp)
        #     return np.array(filtered)
    # ------------------------------------------------------------------------------------------------------------------

    def ignore_specdens(self):
        """
        Removes from the dataset of spectral densities a list of specified methyls that need to be ignored.

        Parameters
        ----------
        idx : list
            indices of methyls to be removed
        """

        # jex = np.empty( (self.jex.shape[0], self.jex.shape[1] - len(self.idx), self.jex.shape[2]) )
        # jmd = np.empty( (self.jmd.shape[0], self.jmd.shape[1] - len(self.idx), self.jmd.shape[2]) )
        # jws = np.empty( (self.jws.shape[0], self.jws.shape[1] - len(self.idx), self.jws.shape[2]) )
        #
        # k = 0
        # for j in range(self.jex.shape[1]):
        #     if j not in self.idx:
        #         jex[:,k,:] = self.jex[:,j,:]
        #         jmd[:,k,:] = self.jmd[:,j,:]
        #         jws[:,k,:] = self.jws[:,j,:]
        #         k += 1
        #
        # self.jex = jex
        # self.jws = jws
        # self.jmd = jmd
        pass

    def load_rotamers():
        """
        Loads into the class the files needed to plot the rotamer distributions

        Parameters
        ----------
        exrot : str
            path to the pickle with the experimental rotamer distributions.
        mdrot : int
            path to the pickle with the simulated rotamer distributions.
        ami : str
            path to the pickle with the amino acid employed in the calculation of the different rotamers.
        """
        pass

        # pin = open(exrot, "rb")
        # self.exrot = pickle.load(pin)
        #
        # pin = open(mdrot, "rb")
        # self.mdrot = pickle.load(pin)
        #
        # pin = open(ami, "rb")
        # self.ami = pickle.load(pin)
        #
        # self.rot_load = True

