########################################################
# Created on June 2, 26
# @author: Johnathan Phillips
# @email: j.s.phillips@wustl.edu

# Purpose: To plot the analyzed data from Apr 25 2026

# Detector: (Insert) scintillator using the CAEN 5730B digitizer

# Channels in use:
#   0 -
########################################################

# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib as mpl
from pathlib import Path
import math
import matplotlib.collections as collections
from matplotlib.legend_handler import HandlerPathCollection
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as colors

# Sets the plot style
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 100
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['xtick.minor.width'] = 1.0
plt.rcParams['xtick.major.size'] = 8.0
plt.rcParams['xtick.minor.size'] = 4.0
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.minor.bottom'] = True
plt.rcParams['xtick.minor.top'] = True
plt.rcParams['xtick.minor.visible'] = True


plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['ytick.minor.width'] = 1.0
plt.rcParams['ytick.major.size'] = 8.0
plt.rcParams['ytick.minor.size'] = 4.0
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.minor.right'] = True
plt.rcParams['ytick.minor.left'] = True
plt.rcParams['ytick.minor.visible'] = True

plt.rcParams['font.size'] = 15
plt.rcParams['axes.titlepad'] = 15

figurepath = Path('OutputFigures') # Go outside of directory and enter Paper_Figures
figurepath = '..' / figurepath

# Linear fit
def linfit(x,a,b):
    return a*x + b

# Inverted linear fit
def invlin(y,a,b):
    return (y - b) / a

def Unpack(name):
    data = np.genfromtxt(name, skip_header = True, delimiter = '\t')
    return data

class MatchedEvent:
    # define dump width map
    dumpWs = [0,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,20,20,20,15,15,15,5,10,10,10,22,22,22,17,17,17,19,19,19]

    # constructor
    def __init__(self, run):
        self.runnum = run
        self.N = [] # number of pulses
        self.CT = [] # cyclotron time
        self.CpC = [] # doseplane charge
        self.CAENt = [] # CAEN t stamp (rel to first hit)
        self.ch0Q = [] # CAEN channel 0 charge
        self.ch0Q_new = []  # CAEN channel 0 charge recalculated
        self.ch2Q = [] # CAEN channel 2 charge
        self.ch2Q_new = []  # CAEN channel 2 charge recalculated
        self.ch4Q = [] # CAEN channel 4 charge
        self.ch4Q_new = []  # CAEN channel 4 charge recalculated
        self.DW = [] # need dump width to get right color map

        self.CpC_avg = 0
        self.CpC_stdev = 0 # standard error of the mean
        self.ch0Q_avg = 0
        self.ch0Q_stdev = 0
        self.ch2Q_avg = 0
        self.ch2Q_stdev = 0
        self.ch4Q_avg = 0
        self.ch4Q_stdev = 0

        self.ch0Q_new_avg = 0
        self.ch0Q_new_stdev = 0
        self.ch2Q_new_avg = 0
        self.ch2Q_new_stdev = 0
        self.ch4Q_new_avg = 0
        self.ch4Q_new_stdev = 0

        # num dumps per chan
        self.Ndumps_0 = 0
        self.Ndumps_2 = 0
        self.Ndumps_4 = 0

    def addevt(self, datarr):
        self.N.append(datarr[0])
        self.CT.append(datarr[1])
        self.CpC.append(datarr[2])
        self.CAENt.append(datarr[3])
        self.ch0Q.append(datarr[4])
        self.ch0Q_new.append(datarr[5])
        self.ch2Q.append(datarr[6])
        self.ch2Q_new.append(datarr[7])
        self.ch4Q.append(datarr[8])
        self.ch4Q_new.append(datarr[9])
        self.DW.append(self.dumpWs[self.runnum])

    # gets averages
    def avgvals(self):

        # filter out zeros first
        if len(self.CpC) > 0 and any(x > 0 for x in self.CpC):
            self.filterdat = [x for x in self.CpC if x > 0]
            self.CpC_avg = np.mean(self.filterdat)
            if len(self.filterdat) > 1: self.CpC_stdev = np.std(self.filterdat, ddof=1) / np.sqrt(len(self.filterdat))
            else: self.CpC_stdev = 0
        else:
            self.CpC_avg = 0
            self.CpC_stdev = 0

        if len(self.ch0Q) > 0 and any(x > 0 for x in self.ch0Q):
            self.filterdat = [x for x in self.ch0Q if x > 0]
            self.ch0Q_avg = np.mean(self.filterdat)
            if len(self.filterdat) > 1:
                self.ch0Q_stdev = np.std(self.filterdat, ddof=1) / np.sqrt(len(self.filterdat))
            else:
                self.ch0Q_stdev = 0
        else:
            self.ch0Q_avg = 0
            self.ch0Q_stdev = 0

        if len(self.ch2Q) > 0 and any(x > 0 for x in self.ch2Q):
            self.filterdat = [x for x in self.ch2Q if x > 0]
            self.ch2Q_avg = np.mean(self.filterdat)
            if len(self.filterdat) > 1:
                self.ch2Q_stdev = np.std(self.filterdat, ddof=1) / np.sqrt(len(self.filterdat))
            else:
                self.ch2Q_stdev = 0
        else:
            self.ch2Q_avg = 0
            self.ch2Q_stdev = 0

        if len(self.ch4Q) > 0 and any(x > 0 for x in self.ch4Q):
            self.filterdat = [x for x in self.ch4Q if x > 0]
            self.ch4Q_avg = np.mean(self.filterdat)
            if len(self.filterdat) > 1:
                self.ch4Q_stdev = np.std(self.filterdat, ddof=1) / np.sqrt(len(self.filterdat))
            else:
                self.ch4Q_stdev = 0
        else:
            self.ch4Q_avg = 0
            self.ch4Q_stdev = 0

    # gets averages for post-processed Qs
    def avgvals_new(self):

        # filter out zeros first
        if len(self.CpC) > 0 and any(x > 0 for x in self.CpC):
            self.filterdat = [x for x in self.CpC if x > 0]
            self.CpC_avg = np.mean(self.filterdat)
            if len(self.filterdat) > 1:
                self.CpC_stdev = np.std(self.filterdat, ddof=1) / np.sqrt(len(self.filterdat))
            else:
                self.CpC_stdev = 0
        else:
            self.CpC_avg = 0
            self.CpC_stdev = 0

        if len(self.ch0Q_new) > 0 and any(x > 0 for x in self.ch0Q_new):
            self.filterdat = [x for x in self.ch0Q_new if x > 0]
            self.ch0Q_new_avg = np.mean(self.filterdat)
            if len(self.filterdat) > 1:
                self.ch0Q_new_stdev = np.std(self.filterdat, ddof=1) / np.sqrt(len(self.filterdat))
            else:
                self.ch0Q_new_stdev = 0
        else:
            self.ch0Q_new_avg = 0
            self.ch0Q_new_stdev = 0

        if len(self.ch2Q_new) > 0 and any(x > 0 for x in self.ch2Q_new):
            self.filterdat = [x for x in self.ch2Q_new if x > 0]
            self.ch2Q_new_avg = np.mean(self.filterdat)
            if len(self.filterdat) > 1:
                self.ch2Q_new_stdev = np.std(self.filterdat, ddof=1) / np.sqrt(len(self.filterdat))
            else:
                self.ch2Q_new_stdev = 0
        else:
            self.ch2Q_new_avg = 0
            self.ch2Q_new_stdev = 0

        if len(self.ch4Q_new) > 0 and any(x > 0 for x in self.ch4Q_new):
            self.filterdat = [x for x in self.ch4Q_new if x > 0]
            self.ch4Q_new_avg = np.mean(self.filterdat)
            if len(self.filterdat) > 1:
                self.ch4Q_new_stdev = np.std(self.filterdat, ddof=1) / np.sqrt(len(self.filterdat))
            else:
                self.ch4Q_new_stdev = 0
        else:
            self.ch4Q_new_avg = 0
            self.ch4Q_new_stdev = 0

    def numdumps(self):
        for i in range(len(self.N)):
            if self.ch0Q_new[i] > 0: self.Ndumps_0 += 1
            if self.ch2Q_new[i] > 0: self.Ndumps_2 += 1
            if self.ch4Q_new[i] > 0: self.Ndumps_4 += 1

def main():
    # Read in analyzed data in the form of txt files
    ###################################################

    # Remember to change your file path for your computer

    ###################################################
    # Constant filepath variable to get around the problem of backslashes in windows
    # The Path library will use forward slashes but convert them to correctly treat your OS
    # Also makes it easier to switch to a different computer

    # file path, use 'Path' function to make compatible with windows/mac
    filepath = Path(r"../Data/Oct3_2026/MatchedEvents")

    rawdat = []
    instances_list = [] # for holding instances of type MatchedEvent

    runnum = 0
    rawdat.append([0]) # append a nothing run
    instances_list.append(MatchedEvent(0))
    # iterate over run numbers
    for i in range(1,35):
        # run 22 is bad
        if i == 22:
            rawdat.append([0])
            instances_list.append(MatchedEvent(i))
            continue
        runnum = i
        filename = r'run' + str(runnum) + r'_match_live.txt'
        file = filepath / filename
        rawdat.append(Unpack(file))

        instances_list.append(MatchedEvent(runnum))

    # Now fill the data in the instances
    for i in range(len(rawdat)):
        if i == 0 or i == 22: continue
        for j in range(len(rawdat[i])):
            instances_list[i].addevt(rawdat[i][j])

    # calculate averages
    for i in range(len(instances_list)):
        if i == 0 or i == 22: continue
        instances_list[i].avgvals()
        instances_list[i].avgvals_new()


    ###################################################
    # Plotting
    chosen_cmap = plt.colormaps.get_cmap('jet')

    # 1st plot: Q vs doseplane IC for runs:
    dumpwidth_runs = [7,8,9,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,33,32,34]
    # 7-9: 750 Hz 25 us
    # 16-18: 750 Hz, 20 us
    # 19-21: 750 Hz, 15 us
    # 23-25: 750 Hz, 10 us
    # 26-28: 750 Hz, 22 us
    # 29-31: 750 Hz, 17 us
    # 32-34: 750 Hz, 19 us
    # Want to colorcode data points according to nominal dump width
    # get list of data that doesn't have any zeros
    allQ0 = []
    allIC_0 = [] # matched with channel 0
    allDWs_0 = []
    for i in dumpwidth_runs:
        if i == 0 or i == 22: continue
        else:
            for j in range(len(instances_list[i].ch0Q)):
                if instances_list[i].ch0Q_new[j] > 0:
                    allIC_0.append(instances_list[i].CpC[j])
                    allQ0.append(instances_list[i].ch0Q_new[j])
                    allDWs_0.append(instances_list[i].DW[j])

    # Get colors
    norm_all = colors.Normalize(vmin=min(allDWs_0), vmax=max(allDWs_0))
    # This generates a list of exact RGBA colors for every data point
    mapped_colors_all = chosen_cmap(norm_all(allDWs_0))

    plot_CvsIC_all0, ax_CvsIC_all0 = plt.subplots()
    ax_CvsIC_all0.scatter(allIC_0, allQ0, color=mapped_colors_all, s=10, label='Channel 0')

    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_CvsIC_all0.colorbar(mappable, ax=ax_CvsIC_all0)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')
    ax_CvsIC_all0.set_xlabel('IC Charge (pC)')
    ax_CvsIC_all0.set_ylabel('CAEN Q (Channels)')
    ax_CvsIC_all0.set_ylim(0,2200)
    ax_CvsIC_all0.set_xlim(0,37.5)

    figgrid_name = 'October3_2026_CAENvsIC_ch0'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

    allQ2 = []
    allIC_2 = [] # matched with channel 2
    allDWs_2 = []
    for i in dumpwidth_runs:
        if i == 0 or i == 22: continue
        else:
            for j in range(len(instances_list[i].ch2Q)):
                if instances_list[i].ch2Q_new[j] > 0:
                    allIC_2.append(instances_list[i].CpC[j])
                    allQ2.append(instances_list[i].ch2Q_new[j])
                    allDWs_2.append(instances_list[i].DW[j])

    # Get colors
    #norm_all = colors.Normalize(vmin=min(allDWs_2), vmax=max(allDWs_2))
    # This generates a list of exact RGBA colors for every data point
    mapped_colors_all = chosen_cmap(norm_all(allDWs_2))

    plot_CvsIC_all2, ax_CvsIC_all2 = plt.subplots()
    ax_CvsIC_all2.scatter(allIC_2, allQ2, color=mapped_colors_all, s=10, label='Channel 2')

    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_CvsIC_all2.colorbar(mappable, ax=ax_CvsIC_all2)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')
    ax_CvsIC_all2.set_xlabel('IC Charge (pC)')
    ax_CvsIC_all2.set_ylabel('CAEN Q (Channels)')
    ax_CvsIC_all2.set_ylim(0,2200)
    ax_CvsIC_all2.set_xlim(0,37.5)

    figgrid_name = 'October3_2026_CAENvsIC_ch2'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

    allQ4 = []
    allIC_4 = [] # matched with channel 4
    allDWs_4 = []
    for i in dumpwidth_runs:
        if i == 0 or i == 22: continue
        else:
            for j in range(len(instances_list[i].ch4Q)):
                if instances_list[i].ch4Q_new[j] > 0:
                    allIC_4.append(instances_list[i].CpC[j])
                    allQ4.append(instances_list[i].ch4Q_new[j])
                    allDWs_4.append(instances_list[i].DW[j])

    # Get colors
    #norm_all = colors.Normalize(vmin=min(allDWs_4), vmax=max(allDWs_4))
    # This generates a list of exact RGBA colors for every data point
    mapped_colors_all = chosen_cmap(norm_all(allDWs_4))

    plot_CvsIC_all4, ax_CvsIC_all4 = plt.subplots()
    ax_CvsIC_all4.scatter(allIC_4, allQ4, color=mapped_colors_all, s=10, label='Channel 4')

    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_CvsIC_all4.colorbar(mappable, ax=ax_CvsIC_all4)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')
    ax_CvsIC_all4.set_xlabel('IC Charge (pC)')
    ax_CvsIC_all4.set_ylabel('CAEN Q (Channels)')
    ax_CvsIC_all4.set_ylim(0,4095)
    ax_CvsIC_all4.set_xlim(0,37.5)

    figgrid_name = 'October3_2026_CAENvsIC_ch4'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

    ###################################################
    # Plot mean and stdev (SEM) of each run in IC and Q
    plot_CvsIC_avg, ax_CvsIC_avg = plt.subplots()
    for i in dumpwidth_runs:
        if i == 0 or i == 22: continue

        run_color = chosen_cmap(norm_all(instances_list[i].DW[0]))

        if instances_list[i].ch0Q_avg > 0:
            ax_CvsIC_avg.scatter(instances_list[i].CpC_avg, instances_list[i].ch0Q_new_avg,color=run_color, s=50, label='Channel 0')
            ax_CvsIC_avg.errorbar(instances_list[i].CpC_avg, instances_list[i].ch0Q_new_avg, xerr=instances_list[i].CpC_stdev, yerr=instances_list[i].ch0Q_new_stdev, fmt='none', elinewidth=1.2, capsize=3, zorder=2,color=run_color)
        if instances_list[i].ch2Q_avg > 0:
            ax_CvsIC_avg.scatter(instances_list[i].CpC_avg, instances_list[i].ch2Q_new_avg, s=50, label='Channel 2',marker='^',color=run_color)
            ax_CvsIC_avg.errorbar(instances_list[i].CpC_avg, instances_list[i].ch2Q_new_avg, xerr=instances_list[i].CpC_stdev,
                                  yerr=instances_list[i].ch2Q_new_stdev, fmt='none', elinewidth=1.2, capsize=3, zorder=2,color=run_color)
        if instances_list[i].ch4Q_avg > 0:
            ax_CvsIC_avg.scatter(instances_list[i].CpC_avg, instances_list[i].ch4Q_new_avg, s=50, label='Channel 2',color=run_color, marker='s')
            ax_CvsIC_avg.errorbar(instances_list[i].CpC_avg, instances_list[i].ch4Q_new_avg, xerr=instances_list[i].CpC_stdev,
                                  yerr=instances_list[i].ch4Q_new_stdev, fmt='none', elinewidth=1.2, capsize=3, zorder=2,color=run_color)

    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_CvsIC_avg.colorbar(mappable, ax=ax_CvsIC_avg)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')
    ax_CvsIC_avg.set_xlabel('IC Charge (pC)')
    ax_CvsIC_avg.set_ylabel('CAEN Q (Channels)')

    figgrid_name = 'October3_2026_AvgCAENvsIC'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

    allQ2 = np.array(allQ2)
    allIC_2 = np.array(allIC_2)

    # Fit the data from channel 2 to a linear
    fit_paramch2, fit_covch2 = curve_fit(linfit, xdata = allIC_2, ydata = allQ2, maxfev = 100000)

    # Get x array
    xspace = np.linspace(0,max(allIC_2), 100)

    # calculate R2
    y_fit = linfit(allIC_2,*fit_paramch2)
    # Residual sum of squares
    ss_res = np.sum((allQ2 - y_fit)**2)
    ss_tot = np.sum((allQ2 - np.mean(allQ2))**2)
    # calc R2
    r_squared = 1 - (ss_res / ss_tot)
    print('ch 2 R2: ', r_squared)

    mapped_colors_all = chosen_cmap(norm_all(allDWs_2))

    plot_CvsIC_all2_fit, ax_CvsIC_all2_fit = plt.subplots()
    ax_CvsIC_all2_fit.scatter(allIC_2, allQ2, color=mapped_colors_all, s=10, label='Channel 2')
    ax_CvsIC_all2_fit.plot(xspace,linfit(xspace,*fit_paramch2),linewidth = 2)

    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_CvsIC_all2_fit.colorbar(mappable, ax=ax_CvsIC_all2_fit)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')
    ax_CvsIC_all2_fit.set_xlabel('IC Charge (pC)')
    ax_CvsIC_all2_fit.set_ylabel('CAEN Q (Channels)')
    ax_CvsIC_all2_fit.set_ylim(0,2200)
    ax_CvsIC_all2_fit.set_xlim(0,37.5)
    figgrid_name = 'October3_2026_ICvsCAEN2_Fit'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

    # Calc Q - Qcalc
    allQ2_diff = allQ2 - linfit(allIC_2, *fit_paramch2)

    plot_allQ2_diff, ax_allQ2_diff = plt.subplots()
    ax_allQ2_diff.hist(allQ2_diff,bins = 50)
    ax_allQ2_diff.set_xlabel('Q - Q$_{calc}$')
    ax_allQ2_diff.set_ylabel(r'Counts')
    plt.show()

    # Now estimate IC from allQ2 and plot hist of ICest / IC
    IC2_est = invlin(allQ2,*fit_paramch2)
    plot_ICRat, ax_ICRat = plt.subplots()
    ax_ICRat.hist(IC2_est/allIC_2,bins = 50)
    ax_ICRat.set_xlabel(r'IC$_{\mathrm{est}}$ / IC')
    ax_ICRat.set_ylabel(r'Counts')
    ax_ICRat.set_xlim(0.2,1.8)
    plt.show()

    # Alternate fit where I ignore points < 8 pC
    # Fit the data from channel 2 to a linear
    gate = allIC_2 >= 8
    allIC_2_disc = allIC_2[gate]
    allQ2_disc = allQ2[gate]

    fit_paramch2, fit_covch2 = curve_fit(linfit, xdata = allIC_2_disc, ydata = allQ2_disc, maxfev = 100000)

    # Get x array
    xspace = np.linspace(0,max(allIC_2), 100)

    mapped_colors_all = chosen_cmap(norm_all(allDWs_2))

    plot_CvsIC_all2_fit, ax_CvsIC_all2_fit = plt.subplots()
    ax_CvsIC_all2_fit.scatter(allIC_2, allQ2, color=mapped_colors_all, s=10, label='Channel 2')
    ax_CvsIC_all2_fit.plot(xspace,linfit(xspace,*fit_paramch2),linewidth = 2)

    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_CvsIC_all2_fit.colorbar(mappable, ax=ax_CvsIC_all2_fit)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')
    ax_CvsIC_all2_fit.set_xlabel('IC Charge (pC)')
    ax_CvsIC_all2_fit.set_ylabel('CAEN Q (Channels)')
    ax_CvsIC_all2_fit.set_ylim(0,2200)
    ax_CvsIC_all2_fit.set_xlim(0,37.5)
    plt.show()

    allQ2 = np.array(allQ2)
    allIC_2 = np.array(allIC_2)

    # Calc Q - Qcalc
    allQ2_diff = allQ2 - linfit(allIC_2, *fit_paramch2)

    plot_allQ2_diff, ax_allQ2_diff = plt.subplots()
    ax_allQ2_diff.hist(allQ2_diff,bins = 50)
    ax_allQ2_diff.set_xlabel('Q - Q$_{calc}$')
    plt.show()

    # Plot CAEN channels against each other

    # ch 0 vs ch 2
    # go through and grab data
    match02_0 = []
    match02_2 = []
    match02_DW = []
    for i in dumpwidth_runs:
        if i == 0 or i == 22: continue
        else:
            for j in range(len(instances_list[i].ch2Q)):
                if instances_list[i].ch2Q_new[j] > 0 and instances_list[i].ch0Q_new[j] > 0:
                    match02_0.append(instances_list[i].ch0Q_new[j])
                    match02_2.append(instances_list[i].ch2Q_new[j])
                    match02_DW.append(instances_list[i].DW[j])

    mapped_colors_all = chosen_cmap(norm_all(match02_DW))
    plot_ch0vs2, ax_ch0vs2 = plt.subplots()
    ax_ch0vs2.scatter(match02_0, match02_2, color=mapped_colors_all, s=10)

    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_ch0vs2.colorbar(mappable, ax=ax_ch0vs2)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')
    ax_ch0vs2.set_xlabel('Channel 0 Q')
    ax_ch0vs2.set_ylabel('Channel 2 Q')
    ax_ch0vs2.set_ylim(0, 2200)
    ax_ch0vs2.set_xlim(0, 2200)
    figgrid_name = 'October3_2026_CAEN0vs2'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()


  # ch 0 vs ch 4
    # go through and grab data
    match04_0 = []
    match04_4 = []
    match04_DW = []
    for i in dumpwidth_runs:
        if i == 0 or i == 22: continue
        else:
            for j in range(len(instances_list[i].ch4Q)):
                if instances_list[i].ch4Q_new[j] > 0 and instances_list[i].ch0Q_new[j] > 0:
                    match04_0.append(instances_list[i].ch0Q_new[j])
                    match04_4.append(instances_list[i].ch4Q_new[j])
                    match04_DW.append(instances_list[i].DW[j])

    mapped_colors_all = chosen_cmap(norm_all(match04_DW))
    plot_ch0vs4, ax_ch0vs4 = plt.subplots()
    ax_ch0vs4.scatter(match04_0, match04_4, color=mapped_colors_all, s=10)

    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_ch0vs4.colorbar(mappable, ax=ax_ch0vs4)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')
    ax_ch0vs4.set_xlabel('Channel 0 Q')
    ax_ch0vs4.set_ylabel('Channel 4 Q')
    ax_ch0vs4.set_ylim(0, 4095)
    ax_ch0vs4.set_xlim(0, 2200)
    figgrid_name = 'October3_2026_CAEN0vs4'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

  # ch 2 vs ch 4
    # go through and grab data
    match24_2 = []
    match24_4 = []
    match24_DW = []
    for i in dumpwidth_runs:
        if i == 0 or i == 22: continue
        else:
            for j in range(len(instances_list[i].ch4Q)):
                if instances_list[i].ch4Q_new[j] > 0 and instances_list[i].ch2Q_new[j] > 0:
                    match24_2.append(instances_list[i].ch2Q_new[j])
                    match24_4.append(instances_list[i].ch4Q_new[j])
                    match24_DW.append(instances_list[i].DW[j])

    mapped_colors_all = chosen_cmap(norm_all(match24_DW))
    plot_ch2vs4, ax_ch2vs4 = plt.subplots()
    ax_ch2vs4.scatter(match24_2, match24_4, color=mapped_colors_all, s=10)

    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_ch2vs4.colorbar(mappable, ax=ax_ch2vs4)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')
    ax_ch2vs4.set_xlabel('Channel 2 Q')
    ax_ch2vs4.set_ylabel('Channel 4 Q')
    ax_ch2vs4.set_ylim(0, 4095)
    ax_ch2vs4.set_xlim(0, 2200)
    figgrid_name = 'October3_2026_CAEN2vs4'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()


    # Get number of digitized pulses per run and average across IC's
    # get number of hits per channel per IC
    plot_Nch0vsDW, ax_Nch0vsD = plt.subplots()
    for i in dumpwidth_runs:
        instances_list[i].numdumps()
        ax_Nch0vsD.scatter(instances_list[i].DW[0], instances_list[i].Ndumps_0, s=30, color='blue',marker='o')
        ax_Nch0vsD.scatter(instances_list[i].DW[0], instances_list[i].Ndumps_2, s=30, color='red',marker='^')
        ax_Nch0vsD.scatter(instances_list[i].DW[0], instances_list[i].Ndumps_4, s=30, color='green',marker='s')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, linestyle='None',
               label='1$^{st}$ Scintillator'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, linestyle='None',
               label='2$^{nd}$ Scintillator'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, linestyle='None',
               label='Optical Fiber'),
    ]

    ax_Nch0vsD.set_ylim(0, 105)
    ax_Nch0vsD.set_xlim(7.5, 27.5)
    ax_Nch0vsD.set_xlabel(r'Dump Width Setting ($\mu$s)')
    ax_Nch0vsD.set_ylabel('Digitized Dumps')
    ax_Nch0vsD.legend(handles=legend_elements, loc='lower right')
    plt.show()




main()