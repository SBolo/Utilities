import numpy as np
import scipy.optimize as optimize
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scipy.stats as scs
from itertools import combinations
import warnings
from . import tools

class Plot():
    def __init__(self):
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['axes.labelsize'] = 18
        mpl.rcParams['legend.fontsize'] = 14

    def phi_chi2r(self, rw, r, outfig=None):

        """
        Plots the phi_eff vs reduced chi squared curve for a provided rate.

        Parameters
        ----------
        r : int
            rate index.
        outfig : str
            path to the figure to save. If not provided, the figure will be prompted on screen and not saved.
        """
        phi, chi = rw.phi_chi2r(r)
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(phi, chi, 'o', c='tab:red', markeredgecolor='k')
        ax.set_ylabel(r'$\chi^2_R$')
        ax.set_xlabel(r'$\phi_{eff}$')
        plt.tight_layout()
        if outfig != None:
            plt.savefig(outfig + '.pdf')
        else:
            plt.show()
    #----------------------------------------------------------------------------------------------------------------

    def comparison(self, data, r, rw=None, theta=None, rate_label=None, outfig=None, show_inset=True):
        """
        Plots the comparison between experimental and simulated data, adding also reweighted data when provided.

        Parameters
        ----------
        data : ABSURDer.data.Data
            ABSURDer data object
        r : int
            rate index
        rw : ABSURDer.reweight.Reweight
            ABSURDer reweighting object
        theta : int
            value of theta corresponding to the reweighting results that want to be shown. If not provided, no
            reweighting results will be shown. rw must be provided as well, otherwise theta will be ignored.
        rate_label : str
            label to print on the y-axis. If not provided, a default label will be printed.
        outfig : str
            path to the figure to save. If not provided, the figure will be prompted on screen and not saved.
        """
        fig, ax = plt.subplots(1, 1, figsize = (7,6.5) )
        if rate_label == None:
            rate_label = r''
        chi_label = r'$\chi^2_R=$'
        theta_label = r'$\theta=$'
        x = np.linspace(-1000,1000,2)
        rates = data.get_rates()
        errors = data.get_errors()
        chi0 = tools.chi2r(rates['experiment'], rates['averaged'], errors['experiment'], errors['simulated'])

        ax.scatter(rates['experiment'][r], rates['averaged'][r], color='k', marker='s',
                   label=f'MD, {chi_label}{chi0:.2f}')
        ax.plot(x, x,'k--', zorder=-1)

        # plot reweighting results only if optimal theta is provided
        if theta != None and rw != None:
            w = rw.results[theta]
            idx = np.argmin(np.fabs(np.array(list(rw.results.keys())) - theta))
            rrw = np.average(rates['simulated'], weights=w, axis=-1)
            phi, chi = rw.phi_chi2r(r)
            ax.scatter(rates['experiment'][r], rrw[r], color='tab:red', marker='o', s=70, alpha=0.8, \
                       label = f'AbsurdER, {theta_label}{theta}, {chi_label}{chi[idx]:.2f}', edgecolor='k')

            if show_inset:
                insax = ax.inset_axes([0.05,0.6,0.4,0.38])
                insax.plot(phi, chi, 'o-', c='tab:grey', markersize=4, mec='k')
                insax.scatter(phi[idx], chi[idx], marker='X', c='tab:red', zorder=10, s=90, edgecolor='k')
                insax.set_xlabel(r'$\phi_{eff}$', fontsize=14)
                insax.set_ylabel(r'$\chi^2_R$', fontsize=14)
                insax.yaxis.tick_right()
                insax.yaxis.set_label_position("right")
                insax.set_xticks([0,0.5,1])
                insax.tick_params(labelsize=14)

        ax.set_xlabel(r'$R^{NMR}$' + rate_label + ' [s$^{-1}$]')
        ax.set_ylabel(r'$R^{SIM}$' + rate_label + ' [s$^{-1}$]')
        ax.set_xlim((0, rates['experiment'][r].max() + 5))
        ax.set_ylim((0, rates['experiment'][r].max() + 5))
        ax.legend(loc=4)
        plt.tight_layout()

        if outfig != None:
            plt.savefig(outfig + '.pdf')
        else:
            plt.show()
    #-------------------------------------------------------------------------------------------------------------------

    def rate_distribution(self, data, methyl_index=None, methyl_name=None, rw=None, theta=None, rate_labels=[], outfig=None):
        """
        Plots the rate distributions over the blocks for a given methyl group.

        Parameters
        ----------
        idx : int
            methyl group index.
        opt_theta : int
            theta corresponding to the optimal set of weights. If not provided, no reweighted results will be shown.
            Default: None.
        methyl_name: str
            name of the methyl group, used as a figure title
        rate_labels : list
            list of labels to print on the x-axis. Is expected to be in a form similar to: [r'(D$_y$)]'. If not provided, a default label will be printed.
            Default: [].
        outfig : str
            path to the figure to save. If not provided, the figure will be prompted on screen and not saved.
            Default = None.
        """
        assert data.get_methyls != [], "No methyl name list provided in the data object."
        fig = plt.figure(figsize=(15, 4))
        if methyl_index == None and methyl_name == None:
            raise ValueError('At least one between methyl_index and methyl_name has to be not None.')
        elif methyl_index == None and methyl_name != None:
            idx = data.methyl_index(methyl_name)
        else:
            idx = methyl_index
            methyl_name = data.get_methyls()[idx]

        rates = data.get_rates()
        errors = data.get_errors()
        for r in range(data.get_nrates()):
            rmd_r_idx = rates['simulated'][r, idx, :]
            x = np.linspace(rmd_r_idx.min(), rmd_r_idx.max(), num=100)
            kde_md = scs.gaussian_kde(rmd_r_idx, bw_method="silverman")
            kde_md.set_bandwidth(kde_md.scotts_factor() / 1.5)
            kde_md = kde_md.evaluate(x)
            if theta != None and rw != None:
                kde_rw = scs.gaussian_kde(rmd_r_idx, bw_method="silverman", weights=rw.results[theta])
                kde_rw.set_bandwidth(kde_rw.scotts_factor() / 1.5)
                kde_rw = kde_rw.evaluate(x)
                rrw = np.average(rates['simulated'], weights=rw.results[theta], axis=-1)
                my = max([max(kde_md), max(kde_rw)])
            else:
                my = max(kde_md)
            my += 0.05 * my

            plt.subplot(1, data.get_nrates(), r + 1)
            plt.plot(x, kde_md, lw=3, color='tab:grey', zorder=-10)
            plt.fill_between(x, kde_md, color='tab:grey', alpha=0.3, label='MD')
            plt.vlines( rates['averaged'][r, idx], 0, my, color='tab:grey', lw=4, linestyle=':', zorder=10,
                        label='Average MD')
            plt.vlines(rates['experiment'][r, idx], 0, my, lw=3, zorder=1, label='NMR')
            plt.axvspan(rates['experiment'][r, idx] - errors['experiment'][r, idx], rates['experiment'][r, idx] + \
                        errors['experiment'][r, idx], 0.05, 0.96, color='k', alpha=0.4, zorder=0)

            if theta != None and rw != None:
                plt.plot(x, kde_rw, lw=3, color='tab:red', zorder=-10)
                plt.fill_between(x, kde_rw, color='tab:red', alpha=0.3, label='ABSURDer')
                plt.vlines(rrw[r, idx], 0, my, color='tab:red', lw=4, linestyle='--', zorder=5,
                           label='Average ABSURDer' )

            if rate_labels == []:
                label = r'Rate [s$^{-1}$]'
            else:
                label = 'R' + rate_labels[r] + r' [s$^{-1}$]'

            plt.xlabel(label)
            if r == 0:
                plt.ylabel( 'p(R)' )
            elif r == data.get_nrates() - 1:
                plt.legend( bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0. )
            plt.suptitle(methyl_name, fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if outfig != None:
                plt.savefig(outfig + '.pdf', format='pdf')
    #-----------------------------------------------------------------------------------------------------------------

    def plot_specdens( self, idx, wd, opt_theta = None, rate_labels = [], outfig = None ):

        """
        Plots the spectral density corresponding to a specific methyl

        Parameters
        ----------
        idx : int
            methyl group index.
        wd: float
            Larmor frequency of 2H at the used magnetic field strength in MHz (Ex. 145.858415 for 2H at 950 MHz magnetic field strength)
        opt_theta : int
            theta corresponding to the optimal set of weights. If not provided, no reweighted results will be shown.
            Default: None.
        methyl_name: str
            name of the methyl group, used as a figure title
        rate_labels : list
            list of labels to print on the x-axis. Is expected to be in a form similar to: [r'(D$_y$)]'. If not provided, a default label will be printed.
            Default: [].
        outfig : str
            path to the figure to save. If not provided, the figure will be prompted on screen and not saved.
            Default = None.
        """

        if not self.specdens_load:
            raise ValueError("Spectral densities have not been loaded. Use load_specdens() for that.")

        pico    = 1. * 10 ** (-12)  # picoseconds
        omega_D = 2. * np.pi * wd * 1000000
        b       = 14  # gridpsec dimension
        wd      = np.array([0, omega_D, 2 * omega_D])
        freq    = np.linspace(0, 2 * 10 ** 9, 100)

        jws = np.average( self.jws, axis = -1 ) / pico
        jex = np.average( self.jex, axis = -1 ) / pico
        jmd = np.average( self.jmd, axis = -1 ) / pico

        fig = plt.figure( figsize = [12, 5], constrained_layout = True)
        gs  = fig.add_gridspec(1, b + self.r)

        ax1 = fig.add_subplot( gs[0, 0:b] )
        ax1.plot( wd,   jws[:, idx],       c = 'k',        ls ='',              zorder=11, marker='v', markersize=10 )
        ax1.plot( freq, jmd[:, idx], lw=4, c = 'tab:grey', ls =':', label='MD', zorder=10)
        ax1.plot( freq, jex[:, idx], lw=3, c = 'k',                 label='NMR')

        if opt_theta != None:
            jrw = np.average( self.jmd, weights = self.res[opt_theta], axis = -1 ) / pico
            rrw = np.average( self.rmd, weights = self.res[opt_theta], axis = -1 )
            ax1.plot( freq, jrw[:, idx], lw = 4, c = 'tab:red', ls = '--', label = 'ABSURDer' )

        ax1.set_ylabel(r'$J$ [ps]')
        ax1.set_xlabel(r'$\omega $ [s$^{-1}$]' + f'\n\n')
        ax1.set_yscale('log')
        ax1.xaxis.offsetText.set_fontsize(14)
        plt.legend( loc = 'upper right' )

        i = b
        j = b + 1
        for r in range(self.r):

            if rate_labels == []:
                rate = 'Rate'
            else:
                rate = 'R' + rate_labels[r]

            ax = fig.add_subplot( gs[0, i:j] )
            ax.errorbar( [rate], self.rex[r, idx], yerr = self.eex[r, idx], elinewidth = 1.2, capthick = 1.2, capsize = 3, marker = 'D',
                         markersize = 10, markeredgecolor = 'k', color = 'k')
            ax.errorbar( [rate], self.rav[r, idx], yerr = self.emd[r, idx], elinewidth = 1.2, capthick = 1.2, capsize = 3, ecolor = 'k', marker = 's',
                        markersize = 10, markeredgecolor = 'k', color = 'tab:grey')

            if opt_theta != None:
                ax.errorbar( [rate], rrw[r, idx], yerr = self.emd[r, idx], elinewidth = 1.2, capthick = 1.2, capsize = 3, ecolor = 'k', marker = 'o',
                            markersize = 11, markeredgecolor = 'k', color = 'tab:red')
            if r == 0:
                ax.set_ylabel(r'Relaxation rate [s$^{-1}$]')

            i += 1
            j += 1

        plt.suptitle(self.mnl[idx], fontsize = 20)

        if outfig != None:
            plt.savefig(outfig + '.pdf', format='pdf')
            print(f"# Saved {outfig}.pdf")
    #-----------------------------------------------------------------------------------------------------------------

    def plot_rotamer_distributions( self, idx, nblocks, block_size, ntrajs, opt_theta = None, outfig = None ):

        """
        Plots the rotamer distributions for a given methyl group.

        Parameters
        ----------
        idx : str
            residue name and number (ex. ILE9).
        nblocks : int
            number of blocks employed in the calculation.
        block_size : int
            size of blocks in ps.
        ntrajs : int
            number of trajectories used to compute the rotamers.
        opt_theta : int
            theta corresponding to the optimal set of weights. If not provided, no reweighted results will be shown.
            Default: None.
        outfig : str
            path to the figure to save. If not provided, the figure will be prompted on screen and not saved.
            Default = None.
        """

        if not self.rot_load:
            raise ValueError("Rotamers have not been loaded. Use load_rotamers() for that.")

        def get_hist(nblocks, blocksize, ang_methyls, mn=-180, mx=180):

            histograms = []
            for b in range(nblocks - 1):
                out = ang_methyls[b * block_size + 1:(b + 1) * block_size]
                h, _ = np.histogram(out, bins=100, range=(mn, mx))
                histograms.append(h)

            return histograms

        chi1      = ['ILE', 'LEU', 'MET', 'THR']
        chi2      = ['ILE', 'LEU', 'MET']
        ang_names = [r"$\chi_1$", r"$\chi_2$", "$\phi$", "$\psi$"]
        rng_max   = [240, 240, 180, 180]
        rng_min   = [-120, -120, -180, -180]
        shift     = [17, 17, 0, 0]
        len_traj  = int(nblocks / ntrajs)

        if idx[:3] in chi2:
            a = 2
            b = 2
            size = (15, 10)
        elif idx[:3] in chi1 and idx[:3] not in chi2:
            a = 1
            b = 3
            size = (15, 5)
        else:
            a = 1
            b = 2
            size = (12, 5)

        fig, axs = plt.subplots(a, b, figsize=size)
        angs = []
        for ax in fig.axes:
            for angg in range(4):
                if idx in self.ami[angg] and angg not in angs:
                    ang = angg
                    angs.append(ang)
                    break

            ind         = self.ami[ang].index(idx)
            tmp_exp     = self.exrot[ang][:, ind]
            hist_exp, _ = np.histogram(tmp_exp, bins=100, range=(-180, 180))
            norm        = np.sum(hist_exp)
            hist_exp    = hist_exp / norm / 3.6
            hist_exp    = np.roll(hist_exp, shift[ang])  # shifts histogram to optimal range

            tmp_md = self.mdrot[ang][:, ind]
            hist   = get_hist( nblocks, block_size, tmp_md )

            conc = ()
            for n in range(1, ntrajs + 1):
                conc = conc + (hist[(n - 1) * len_traj:n * len_traj - 1],)
            hist = np.concatenate(conc)

            hist_sum = np.average(hist, axis=0) * len(hist)
            norm     = np.sum(hist_sum)
            hist_md = hist_sum / norm / 3.6
            hist_md = np.roll(hist_md, shift[ang])

            if opt_theta != None:
                hist_sum = np.average(hist, axis=0, weights=self.res[opt_theta]) * len(hist)
                norm     = np.sum(hist_sum)
                hist_rw  = hist_sum / norm / 3.6
                hist_rw  = np.roll(hist_rw, shift[ang])

            ax.plot(np.linspace(rng_min[ang], rng_max[ang], 100), hist_exp, c='k', lw=4, label='NMR')
            ax.plot(np.linspace(rng_min[ang], rng_max[ang], 100), hist_md, c='tab:grey', lw=4, ls=':', label='MD')

            if opt_theta != None:
                ax.plot(np.linspace(rng_min[ang], rng_max[ang], 100), hist_rw, c='tab:red', lw=4, ls='--', label='ABSURDer')
            ax.set_xlabel(ang_names[ang])
            ax.set_ylabel(None)
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:1.2f}'))

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        fig.suptitle(idx, fontsize=22)

        if outfig != None:
            plt.savefig(outfig + '.pdf', format='pdf')
            print(f"# Saved {outfig}.pdf")

    #------------------------------------------------------------------------------------------------------------------

    def plot_single_rotamer( self, idx, ang, nblocks, block_size, ntrajs, opt_theta = None, outfig = None ):

        """
        Plots the rotamer distributions for a given methyl group.

        Parameters
        ----------
        idx : str
            residue name and number (ex. ILE9).
        nblocks : int
            number of blocks employed in the calculation.
        block_size : int
            size of blocks in ps.
        ntrajs : int
            number of trajectories used to compute the rotamers.
        opt_theta : int
            theta corresponding to the optimal set of weights. If not provided, no reweighted results will be shown.
            Default: None.
        outfig : str
            path to the figure to save. If not provided, the figure will be prompted on screen and not saved.
            Default = None.
        """

        def get_hist(nblocks, blocksize, ang_methyls, mn=-180, mx=180):

            histograms = []
            for b in range(nblocks - 1):
                out = ang_methyls[b * block_size + 1:(b + 1) * block_size]
                h, _ = np.histogram(out, bins=100, range=(mn, mx))
                histograms.append(h)

            return histograms

        chi1      = ['ILE', 'LEU', 'MET', 'THR']
        chi2      = ['ILE', 'LEU', 'MET']
        ang_names = [r"$\chi_1$", r"$\chi_2$", "$\phi$", "$\psi$"]
        rng_max   = [240, 240, 180, 180]
        rng_min   = [-120, -120, -180, -180]
        shift     = [17, 17, 0, 0]
        len_traj  = int(nblocks / ntrajs)

        plt.figure( figsize = ( 9.55, 5 ) )
        plt.title( idx, fontsize=18, weight = 'bold')
        angs = []

        ind         = self.ami[ang].index(idx)
        tmp_exp     = self.exrot[ang][:, ind]
        hist_exp, _ = np.histogram(tmp_exp, bins=100, range=(-180, 180))
        norm        = np.sum(hist_exp)
        hist_exp    = hist_exp / norm / 3.6
        hist_exp    = np.roll(hist_exp, shift[ang])  # shifts histogram to optimal range

        tmp_md = self.mdrot[ang][:, ind]
        hist   = get_hist( nblocks, block_size, tmp_md )

        conc = ()
        for n in range(1, ntrajs + 1):
            conc = conc + (hist[(n - 1) * len_traj:n * len_traj - 1],)
        hist = np.concatenate(conc)

        hist_sum = np.average(hist, axis=0) * len(hist)
        norm     = np.sum(hist_sum)
        hist_md = hist_sum / norm / 3.6
        hist_md = np.roll(hist_md, shift[ang])

        if opt_theta != None:
            hist_sum = np.average(hist, axis=0, weights=self.res[opt_theta]) * len(hist)
            norm     = np.sum(hist_sum)
            hist_rw  = hist_sum / norm / 3.6
            hist_rw  = np.roll(hist_rw, shift[ang])

        plt.plot(np.linspace(rng_min[ang], rng_max[ang], 100), hist_exp, c='k', lw=4, label='NMR')
        plt.plot(np.linspace(rng_min[ang], rng_max[ang], 100), hist_md, c='tab:grey', lw=4, ls=':', label='MD')

        if opt_theta != None:
            plt.plot(np.linspace(rng_min[ang], rng_max[ang], 100), hist_rw, c='tab:red', lw=4, ls='--', label='ABSURDer')
        plt.xlabel(ang_names[ang])
        plt.ylabel(r'$p($' + ang_names[ang] + r'$)$')
        #ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:1.2f}'))

        if outfig != None:
            plt.tight_layout()
            plt.savefig(outfig + '.pdf', format='pdf')
            print(f"# Saved {outfig}.pdf")

    #------------------------------------------------------------------------------------------------------------------

    def plot_2d_rotamers( self, idx, nblocks, block_size, ntrajs, opt_theta, outfig = None ):

        """
        Plots the chi1-chi2 rotamer distribution for a given methyl group.

        Parameters
        ----------
        idx : str
            residue name and number (ex. ILE9).
        nblocks: int
            number of blocks employed in the calculation.
        block_size : int
            size of blocks in ps.
        opt_theta : int
            theta corresponding to the optimal set of weights. If not provided, no reweighted results will be shown.
        outfig : str
            path to the figure to save. If not provided, the figure will be prompted on screen and not saved.
            Default = None.
        """

        if not self.rot_load:
            raise ValueError("Rotamers have not been loaded. Use load_rotamers() for that.")

        ind         = self.ami[0].index(idx)
        ind2        = self.ami[1].index(idx)
        tmp_md_chi1 = self.mdrot[0][:, ind]
        tmp_md_chi2 = self.mdrot[1][:, ind2]
        len_traj    = int(nblocks / ntrajs)

        hist = []
        for b in range(nblocks - 1):
            out_chi1 = tmp_md_chi1[b * block_size + 1:(b + 1) * block_size]
            out_chi2 = tmp_md_chi2[b * block_size + 1:(b + 1) * block_size]
            h, _, _ = np.histogram2d(out_chi1, out_chi2, bins=100, range=[[-180, 180], [-180, 180]])
            hist.append(h)

        conc = ()
        for n in range(1, ntrajs + 1):
            conc = conc + (hist[(n - 1) * len_traj:n * len_traj - 1],)
        hist = np.concatenate(conc)

        hist_md = np.average(hist, axis=0) * len(hist)
        hist_md = np.roll(hist_md, 17, axis=0)
        hist_md = np.roll(hist_md, 17, axis=1)

        norm = np.sum(hist_md)
        hist_md = hist_md / norm / 3.6 / 3.6

        chi1_exp = self.exrot[0][:, ind]
        chi2_exp = self.exrot[1][:, ind2]

        h, _, _ = np.histogram2d(chi1_exp, chi2_exp, bins=100, range=[[-180, 180], [-180, 180]])
        norm = np.sum(h)
        h = h / norm / 3.6 / 3.6
        h = np.roll(h, 17, axis=0)
        h = np.roll(h, 17, axis=1)

        hist_rw = np.average(hist, axis=0, weights=self.res[opt_theta]) * len(hist)
        hist_rw = np.roll(hist_rw, 17, axis=0)
        hist_rw = np.roll(hist_rw, 17, axis=1)

        norm = np.sum(hist_rw)
        hist_rw = hist_rw / norm / 3.6 / 3.6

        plt.figure(figsize=(19, 6))

        oldcmp = mpl.cm.get_cmap('Reds', 512)
        newcmp = ListedColormap(oldcmp(np.linspace(0, 0.75, 384)))

        plt.subplot(1, 3, 1)
        plt.contourf(hist_md.T, 50, cmap=newcmp, zorder=1, origin='lower', extent=(-120, 240, -120, 240),
                     vmax=8e-4)
        plt.contour(hist_md.T, levels=np.arange(0, 8e-4, 1e-4), colors='k', linewidths=0.6, zorder=10, origin='lower',
                    extent=(-120, 240, -120, 240))
        plt.title('MD', fontsize=18)
        plt.xlabel(r'$\chi_1$ [deg]')
        plt.ylabel(r'$\chi_2$ [deg]')

        plt.subplot(1, 3, 2)
        plt.contourf(hist_rw.T, 50, cmap=newcmp, zorder=1, origin='lower',
                     extent=(-120, 240, -120, 240), vmax=8e-4)
        plt.contour(hist_rw.T, levels=np.arange(0, 8e-4, 1e-4), colors='k', linewidths=0.6, zorder=10, origin='lower',
                    extent=(-120, 240, -120, 240))
        plt.title('ABSURDer', fontsize=18)
        plt.xlabel(r'$\chi_1$ [deg]')

        plt.subplot(1, 3, 3)
        plt.contourf(h.T, 50, cmap=newcmp, zorder=1, origin='lower', extent=(-120, 240, -120, 240), vmax=8e-4)

        m = mpl.cm.ScalarMappable(cmap=newcmp)
        m.set_array(h)
        m.set_clim(0., 8e-4)
        cbar = plt.colorbar(m, boundaries=np.linspace(0, 8e-4, 100), ticks=[0, 2e-4, 4e-4, 6e-4, 8e-4])
        cbar.ax.set_xticklabels(
            [0, r'$2\times 10^{-4}$', r'$4\times 10^{-4}$', r'$6\times 10^{-4}$', r'$8 \times 10^{-4}$'])
        cbar.set_label('Probability Density')

        plt.contour(h.T, levels=np.arange(0, 8e-4, 1e-4), colors='k', linewidths=0.6, zorder=10, origin='lower',
                    extent=(-120, 240, -120, 240))
        plt.title('NMR', fontsize=18)
        plt.xlabel(r'$\chi_1$ [deg]')

        plt.suptitle(idx, fontsize=18, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        return hist_md, hist_rw, h
    #------------------------------------------------------------------------------------------------------------------

    def phi_psi_rmsd( self, ang, nblocks, block_size, ntrajs, opt_theta ):

        def get_hist(nblocks, blocksize, ang_methyls, mn=-180, mx=180):

            histograms = []
            for b in range(nblocks - 1):
                out = ang_methyls[b * block_size + 1:(b + 1) * block_size]
                h, _ = np.histogram(out, bins = 100, range=(mn, mx))
                histograms.append(h)

            return histograms

        len_traj = int(nblocks / ntrajs)
        rmsd = []

        for res in self.ami[ang]:
            ind = self.ami[ang].index(res)
            tmp_exp = self.exrot[ang][:, ind]
            hist_exp, _ = np.histogram(tmp_exp, bins=100, range=(-180, 180))
            norm = np.sum(hist_exp)
            hist_exp = hist_exp / norm / 3.6

            tmp_md = self.mdrot[ang][:, ind]
            hist = get_hist(nblocks, block_size, tmp_md)

            conc = ()
            for n in range(1, ntrajs + 1):
                conc = conc + (hist[(n - 1) * len_traj:n * len_traj - 1],)
            hist = np.concatenate(conc)

            hist_sum = np.average(hist, axis=0) * len(hist)
            norm = np.sum(hist_sum)
            hist_md = hist_sum / norm / 3.6

            hist_sum = np.average(hist, axis=0, weights = self.res[opt_theta]) * len(hist)
            norm = np.sum(hist_sum)
            hist_rw = hist_sum / norm / 3.6

            rmsd_md = self.rmsd( hist_exp, hist_md )
            rmsd_rw = self.rmsd( hist_exp, hist_rw )
            rmsd.append( (rmsd_md - rmsd_rw) )

        return rmsd
    #------------------------------------------------------------------------------------------------------------------

    def rmsd( self, exp, md ):
        rmsd = np.sqrt( 1 / len(exp) * np.sum( ( exp - md )**2 ) )
        return rmsd
    #------------------------------------------------------------------------------------------------------------------

    def plot_delta_rmsds( self, ang, delta, label, outfig = None ):

        mpl.rcParams['xtick.labelsize'] = 18
        mpl.rcParams['ytick.labelsize'] = 18

        palette = []
        for r in self.ami[ang]:
            if 'ALA' in r:
                palette.append('tab:red')
            elif 'ILE' in r:
                palette.append('tab:brown')
            elif 'LEU' in r:
                palette.append('tab:green')
            elif 'THR' in r:
                palette.append('tab:orange')
            elif 'VAL' in r:
                palette.append('tab:blue')
            elif 'MET' in r:
                palette.append('tab:purple')

        custom_lines = [Patch(edgecolor='k', facecolor='tab:red'),
                        Patch(edgecolor='k', facecolor='tab:brown'),
                        Patch(edgecolor='k', facecolor='tab:green'),
                        Patch(edgecolor='k', facecolor='tab:orange'),
                        Patch(edgecolor='k', facecolor='tab:blue'),
                        Patch(edgecolor='k', facecolor='tab:purple')]

        labels = ['ALA', 'ILE', 'LEU', 'THR', 'VAL', 'MET']

        fig = plt.figure( figsize=(9.55, 6) )

        plt.bar( np.arange(0,len(delta),1), delta, edgecolor='k', color = palette, zorder=10 )

        plt.xlabel('Residues' )
        plt.ylabel(r'$\Delta $RMSD(' + label + ')' )
        plt.tight_layout()

        if outfig != None:
            plt.savefig( outfig + '.pdf', format = 'pdf')
        else:
            plt.show()
    #------------------------------------------------------------------------------------------------------------------
