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

def main():
    # Read in analyzed data in the form of txt files
    ###################################################

    # Remember to change your file path for your computer

    ###################################################
    # Constant filepath variable to get around the problem of backslashes in windows
    # The Path library will use forward slashes but convert them to correctly treat your OS
    # Also makes it easier to switch to a different computer

    # file path, use 'Path' function to make compatible with windows/mac
    filepath = Path(r"../Data/April25_2026")

    # Set 1 average values (different beam widths)
    S1avgfilen = r'April25_26_1stSetAvg.txt'
    S1avgfile = filepath / S1avgfilen

    # Set 2 average values (duplicates of 1)
    S2avgfilen = r'April25_26_2ndSetAvg.txt'
    S2avgfile = filepath / S2avgfilen

    # Set 1 all data (odd runs 1-13)
    S1allfilen = r'April25_26_1stSetall.txt'
    S1allfile = filepath / S1allfilen

    # Set 2 all data (even runs 2-14)
    S2allfilen = r'April25_26_2ndSetall.txt'
    S2allfile = filepath / S2allfilen

    # Unpack data using genfromtxt, sorted by run #
    S1avgdata = np.genfromtxt(S1avgfile, skip_header = True, max_rows = 50000)
    S2avgdata = np.genfromtxt(S2avgfile, skip_header = True, max_rows = 50000)

    S1alldata = np.genfromtxt(S1allfile, skip_header = True, max_rows = 50000)
    S2alldata = np.genfromtxt(S2allfile, skip_header = True, max_rows = 50000)

    ##############################################################################
    #Plotting average data

    S1run = [] # run num
    S1dwidth = [] # dump width
    S1ic = [] # ion chamber charge (pC, all vals)
    S1ic_e = [] # standard error of mean for ic average
    S1caen_0 = [] # CAEN Q (Channels) from scintillator channel 0
    S1caen_e_0 = [] # sem for CAEN channel 0
    S1caen_1 = [] # CAEN Q (Channels) from scintillator channel 1
    S1caen_e_1 = [] # sem for CAEN channel 1
    S1caen_2 = [] # CAEN Q (Channels) from scintillator channel 1
    S1caen_e_2 = [] # sem for CAEN channel 1
    S1ic_0 = [] # ion chamber charge (pC, match with CAEN ch 0)
    S1ic_e_0 = [] # standard error of mean for ic match with CAEN ch 0
    S1ic_1 = [] # ion chamber charge (pC, match with CAEN ch 1)
    S1ic_e_1 = [] # standard error of mean for ic match with CAEN ch 1
    S1ic_2 = [] # ion chamber charge (pC, match with CAEN ch 2)
    S1ic_e_2 = [] # standard error of mean for ic match with CAEN ch 2

    for i in range(len(S1avgdata)):
        S1run.append(S1avgdata[i][0])
        S1dwidth.append(S1avgdata[i][1])
        S1ic.append(S1avgdata[i][2])
        S1ic_e.append(S1avgdata[i][3])
        S1caen_0.append(S1avgdata[i][4])
        S1caen_e_0.append(S1avgdata[i][5])
        S1caen_1.append(S1avgdata[i][6])
        S1caen_e_1.append(S1avgdata[i][7])
        S1caen_2.append(S1avgdata[i][8])
        S1caen_e_2.append(S1avgdata[i][9])
        S1ic_0.append(S1avgdata[i][10])
        S1ic_e_0.append(S1avgdata[i][11])
        S1ic_1.append(S1avgdata[i][12])
        S1ic_e_1.append(S1avgdata[i][13])
        S1ic_2.append(S1avgdata[i][14])
        S1ic_e_2.append(S1avgdata[i][15])

    S1run = np.array(S1run)
    S1dwidth = np.array(S1dwidth)
    S1ic = np.array(S1ic)
    S1ic_e = np.array(S1ic_e)
    S1caen_0 = np.array(S1caen_0)
    S1caen_e_0 = np.array(S1caen_e_0)
    S1caen_1 = np.array(S1caen_1)
    S1caen_e_1 = np.array(S1caen_e_1)
    S1caen_2 = np.array(S1caen_2)
    S1caen_e_2 = np.array(S1caen_e_2)
    S1ic_0 = np.array(S1ic_0)
    S1ic_e_0 = np.array(S1ic_e_0)
    S1ic_1 = np.array(S1ic_1)
    S1ic_e_1 = np.array(S1ic_e_1)
    S1ic_2 = np.array(S1ic_2)
    S1ic_e_2 = np.array(S1ic_e_2)

    S2run = []  # run num
    S2dwidth = []  # dump width
    S2ic = []  # ion chamber charge (pC, all vals)
    S2ic_e = []  # standard error of mean for ic average
    S2caen_0 = []  # CAEN Q (Channels) from scintillator channel 0
    S2caen_e_0 = []  # sem for CAEN channel 0
    S2caen_1 = []  # CAEN Q (Channels) from scintillator channel 1
    S2caen_e_1 = []  # sem for CAEN channel 1
    S2caen_2 = []  # CAEN Q (Channels) from scintillator channel 1
    S2caen_e_2 = []  # sem for CAEN channel 1
    S2ic_0 = []  # ion chamber charge (pC, match with CAEN ch 0)
    S2ic_e_0 = []  # standard error of mean for ic match with CAEN ch 0
    S2ic_1 = []  # ion chamber charge (pC, match with CAEN ch 1)
    S2ic_e_1 = []  # standard error of mean for ic match with CAEN ch 1
    S2ic_2 = []  # ion chamber charge (pC, match with CAEN ch 2)
    S2ic_e_2 = []  # standard error of mean for ic match with CAEN ch 2

    for i in range(len(S2avgdata)):
        S2run.append(S2avgdata[i][0])
        S2dwidth.append(S2avgdata[i][1])
        S2ic.append(S2avgdata[i][2])
        S2ic_e.append(S2avgdata[i][3])
        S2caen_0.append(S2avgdata[i][4])
        S2caen_e_0.append(S2avgdata[i][5])
        S2caen_1.append(S2avgdata[i][6])
        S2caen_e_1.append(S2avgdata[i][7])
        S2caen_2.append(S2avgdata[i][8])
        S2caen_e_2.append(S2avgdata[i][9])
        S2ic_0.append(S2avgdata[i][10])
        S2ic_e_0.append(S2avgdata[i][11])
        S2ic_1.append(S2avgdata[i][12])
        S2ic_e_1.append(S2avgdata[i][13])
        S2ic_2.append(S2avgdata[i][14])
        S2ic_e_2.append(S2avgdata[i][15])

    S2run = np.array(S2run)
    S2dwidth = np.array(S2dwidth)
    S2ic = np.array(S2ic)
    S2ic_e = np.array(S2ic_e)
    S2caen_0 = np.array(S2caen_0)
    S2caen_e_0 = np.array(S2caen_e_0)
    S2caen_1 = np.array(S2caen_1)
    S2caen_e_1 = np.array(S2caen_e_1)
    S2caen_2 = np.array(S2caen_2)
    S2caen_e_2 = np.array(S2caen_e_2)
    S2ic_0 = np.array(S2ic_0)
    S2ic_e_0 = np.array(S2ic_e_0)
    S2ic_1 = np.array(S2ic_1)
    S2ic_e_1 = np.array(S2ic_e_1)
    S2ic_2 = np.array(S2ic_2)
    S2ic_e_2 = np.array(S2ic_e_2)

    # Plot average values as CAEN (Y) vs IC (X)
    # Change point color depending on value of dwidth (blue to red)
    # Plot of energies vs Time (raw)
    plot_CvsIC_0, ax_CvsIC_0 = plt.subplots()
    chosen_cmap = plt.colormaps.get_cmap('jet')
    # Normalize maps your dump widths between a 0.0 and 1.0 scale
    norm = colors.Normalize(vmin=min(S2dwidth), vmax=max(S2dwidth))
    # This generates a list of exact RGBA colors for every data point
    mapped_colors = chosen_cmap(norm(S2dwidth))

    scatter_S1_0 = ax_CvsIC_0.scatter(S1ic_0,S1caen_0,c=mapped_colors, s=50, label='Set 1')
    # Loop through each point in Set 1 to apply its specific color to its error bars
    for i in range(len(S1caen_0)):
        ax_CvsIC_0.errorbar(S1ic_0[i],S1caen_0[i], xerr=S1ic_e_0[i], yerr=S1caen_e_0[i], fmt='none',
                          ecolor=mapped_colors[i], elinewidth=1.2, capsize=3, zorder=2)

    scatter_S2_0 = ax_CvsIC_0.scatter( S2ic_0,S2caen_0,marker='^',c=mapped_colors, s=50, label='Set 2')
    # Loop through each point in Set 2 to apply its specific color to its error bars
    for i in range(len(S2caen_0)):
        ax_CvsIC_0.errorbar( S2ic_0[i],S2caen_0[i], xerr=S2ic_e_0[i], yerr=S2caen_e_0[i], fmt='none',
                          ecolor=mapped_colors[i], elinewidth=1.2, capsize=3, zorder=2)

    mappable = cm.ScalarMappable(norm=norm, cmap=chosen_cmap)
    cbar = plot_CvsIC_0.colorbar(mappable, ax=ax_CvsIC_0)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')

    ax_CvsIC_0.set_xlabel('Average IC Charge (pC)')
    ax_CvsIC_0.set_ylabel('Average CAEN Q (Channels)')
    ax_CvsIC_0.set_xlim(-0.5,16)

    # Makes the legend markers filled and outlined with black
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 1'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 2')
    ]
    ax_CvsIC_0.legend(handles=legend_handles)

    figgrid_name = 'April25_2026_AvgCAENvsIC_ch0'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

    plot_CvsIC_1, ax_CvsIC_1 = plt.subplots()

    mask = (S1ic_1 != 0) & (S1caen_1 != 0)

    scatter_S1_1 = ax_CvsIC_1.scatter(S1ic_1[mask], S1caen_1[mask], c=mapped_colors[mask], s=50, label='Set 1')
    # Loop through each point in Set 1 to apply its specific color to its error bars
    for i in range(len(S1caen_1)):
        if S1ic_1[i] > 0: ax_CvsIC_1.errorbar(S1ic_1[i], S1caen_1[i], xerr=S1ic_e_1[i], yerr=S1caen_e_1[i], fmt='none',
                            ecolor=mapped_colors[i], elinewidth=1.2, capsize=3, zorder=2)

    scatter_S2_1 = ax_CvsIC_1.scatter(S2ic_1, S2caen_1, marker='^', c=mapped_colors, s=50, label='Set 2')
    # Loop through each point in Set 2 to apply its specific color to its error bars
    for i in range(len(S2caen_1)):
        ax_CvsIC_1.errorbar(S2ic_1[i], S2caen_1[i], xerr=S2ic_e_1[i], yerr=S2caen_e_1[i], fmt='none',
                            ecolor=mapped_colors[i], elinewidth=1.2, capsize=3, zorder=2)

    mappable = cm.ScalarMappable(norm=norm, cmap=chosen_cmap)
    cbar = plot_CvsIC_1.colorbar(mappable, ax=ax_CvsIC_1)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')

    ax_CvsIC_1.set_xlabel('Average IC Charge (pC)')
    ax_CvsIC_1.set_ylabel('Average CAEN Q (Channels)')
    ax_CvsIC_1.set_xlim(-0.5,16)

    # Makes the legend markers filled and outlined with black
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 1'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 2')
    ]
    ax_CvsIC_1.legend(handles=legend_handles)

    figgrid_name = 'April25_2026_AvgCAENvsIC_ch1'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

    plot_CvsIC_2, ax_CvsIC_2 = plt.subplots()

    mask = (S1ic_2 != 0) & (S1caen_2 != 0)
    mask2 = (S2ic_2 != 0) & (S2caen_2 != 0)


    scatter_S1_2 = ax_CvsIC_2.scatter(S1ic_2[mask], S1caen_2[mask], c=mapped_colors[mask], s=50, label='Set 1')
    # Loop through each point in Set 1 to apply its specific color to its error bars
    for i in range(len(S1caen_2)):
        if S1ic_2[i] > 0: ax_CvsIC_2.errorbar(S1ic_2[i], S1caen_2[i], xerr=S1ic_e_2[i], yerr=S1caen_e_2[i], fmt='none',
                            ecolor=mapped_colors[i], elinewidth=1.2, capsize=3, zorder=2)

    scatter_S2_2 = ax_CvsIC_2.scatter(S2ic_2[mask2], S2caen_2[mask2], marker='^', c=mapped_colors[mask2], s=50, label='Set 2')
    # Loop through each point in Set 2 to apply its specific color to its error bars
    for i in range(len(S2caen_2)):
        if S2ic_2[i] > 0: ax_CvsIC_2.errorbar(S2ic_2[i], S2caen_2[i], xerr=S2ic_e_2[i], yerr=S2caen_e_2[i], fmt='none',
                            ecolor=mapped_colors[i], elinewidth=1.2, capsize=3, zorder=2)

    mappable = cm.ScalarMappable(norm=norm, cmap=chosen_cmap)
    cbar = plot_CvsIC_2.colorbar(mappable, ax=ax_CvsIC_2)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')

    ax_CvsIC_2.set_xlabel('Average IC Charge (pC)')
    ax_CvsIC_2.set_ylabel('Average CAEN Q (Channels)')
    ax_CvsIC_2.set_xlim(-0.5,16)
    ax_CvsIC_2.set_ylim(0,345)

    # Makes the legend markers filled and outlined with black
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 1'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 2')
    ]
    ax_CvsIC_2.legend(handles=legend_handles)

    figgrid_name = 'April25_2026_AvgCAENvsIC_ch2'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()


    # Plot both ch 0 and 1 on the same plot
    plot_CvsIC_01, ax_CvsIC_01 = plt.subplots()

    scatter_S1_0 = ax_CvsIC_01.scatter(S1ic_0, S1caen_0, c=mapped_colors, s=50, label='Set 1')
    # Loop through each point in Set 1 to apply its specific color to its error bars
    for i in range(len(S1caen_0)):
        ax_CvsIC_01.errorbar(S1ic_0[i], S1caen_0[i], xerr=S1ic_e_0[i], yerr=S1caen_e_0[i], fmt='none',
                            ecolor=mapped_colors[i], elinewidth=1.2, capsize=3, zorder=2)

    scatter_S2_0 = ax_CvsIC_01.scatter(S2ic_0, S2caen_0, marker='^', c=mapped_colors, s=50, label='Set 2')
    # Loop through each point in Set 2 to apply its specific color to its error bars
    for i in range(len(S2caen_0)):
        ax_CvsIC_01.errorbar(S2ic_0[i], S2caen_0[i], xerr=S2ic_e_0[i], yerr=S2caen_e_0[i], fmt='none',
                            ecolor=mapped_colors[i], elinewidth=1.2, capsize=3, zorder=2)

    mask = (S1ic_1 != 0) & (S1caen_1 != 0)

    scatter_S1_1 = ax_CvsIC_01.scatter(S1ic_1[mask], S1caen_1[mask], c=mapped_colors[mask], s=50, label='Set 1')
    # Loop through each point in Set 1 to apply its specific color to its error bars
    for i in range(len(S1caen_1)):
        if S1ic_1[i] > 0: ax_CvsIC_01.errorbar(S1ic_1[i], S1caen_1[i], xerr=S1ic_e_1[i], yerr=S1caen_e_1[i], fmt='none',
                                       ecolor=mapped_colors[i], elinewidth=1.2, capsize=3, zorder=2)

    scatter_S2_1 = ax_CvsIC_01.scatter(S2ic_1, S2caen_1, marker='^', c=mapped_colors, s=50, label='Set 2')
    # Loop through each point in Set 2 to apply its specific color to its error bars
    for i in range(len(S2caen_1)):
        ax_CvsIC_01.errorbar(S2ic_1[i], S2caen_1[i], xerr=S2ic_e_1[i], yerr=S2caen_e_1[i], fmt='none',
                            ecolor=mapped_colors[i], elinewidth=1.2, capsize=3, zorder=2)

    mappable = cm.ScalarMappable(norm=norm, cmap=chosen_cmap)
    cbar = plot_CvsIC_01.colorbar(mappable, ax=ax_CvsIC_01)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')

    ax_CvsIC_01.set_xlabel('Average IC Charge (pC)')
    ax_CvsIC_01.set_ylabel('Average CAEN Q (Channels)')
    ax_CvsIC_01.set_xlim(-0.5, 16)

    # Makes the legend markers filled and outlined with black
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 1'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 2')
    ]
    ax_CvsIC_01.legend(handles=legend_handles)

    figgrid_name = 'April25_2026_AvgCAENvsIC_ch01'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()


    ##################################################################

    ##################################################################
    #Plotting all data
    S1all_run = []
    S1all_dwidth = []
    S1all_ic = []
    S1all_caen_0 = []
    S1all_caen_1 = []
    S1all_caen_2 = []
    for i in range(len(S1alldata)):
        S1all_run.append(S1alldata[i][0])
        S1all_dwidth.append(S1alldata[i][1])
        S1all_ic.append(S1alldata[i][2])
        S1all_caen_0.append(S1alldata[i][3])
        S1all_caen_1.append(S1alldata[i][4])
        S1all_caen_2.append(S1alldata[i][5])

    S2all_run = []
    S2all_dwidth = []
    S2all_ic = []
    S2all_caen_0 = []
    S2all_caen_1 = []
    S2all_caen_2 = []
    for i in range(len(S2alldata)):
        S2all_run.append(S2alldata[i][0])
        S2all_dwidth.append(S2alldata[i][1])
        S2all_ic.append(S2alldata[i][2])
        S2all_caen_0.append(S2alldata[i][3])
        S2all_caen_1.append(S2alldata[i][4])
        S2all_caen_2.append(S2alldata[i][5])

    # Big plot time
    # Want to do IC (x) vs CAEN (y)
    # Also want to omit points with CAEN = 0, so do loop scatter

    # Get colors
    norm_all = colors.Normalize(vmin=min(S1all_dwidth), vmax=max(S1all_dwidth))
    # This generates a list of exact RGBA colors for every data point
    mapped_colors_all = chosen_cmap(norm(S1all_dwidth))

    plot_CvsIC_all0, ax_CvsIC_all0 = plt.subplots()
    for i in range(len(S1all_ic)):
        if S1all_caen_0[i] > 0: ax_CvsIC_all0.scatter(S1all_ic[i], S1all_caen_0[i], color=mapped_colors_all[i], s=25, label='Set 1')

    for i in range(len(S2all_ic)):
        if S2all_caen_0[i] > 0: ax_CvsIC_all0.scatter(S2all_ic[i], S2all_caen_0[i], color=mapped_colors_all[i], marker='^', s=25, label='Set 1')


    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_CvsIC_all0.colorbar(mappable, ax=ax_CvsIC_all0)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')

    ax_CvsIC_all0.set_xlabel('IC Charge (pC)')
    ax_CvsIC_all0.set_ylabel('CAEN Q (Channels)')
    ax_CvsIC_all0.set_xlim(-1, 20)
    ax_CvsIC_all0.set_ylim(750, 2100)

    # Makes the legend markers filled and outlined with black
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 1'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 2')
    ]
    ax_CvsIC_all0.legend(handles=legend_handles,loc='upper left')

    figgrid_name = 'April25_2026_CAENvsIC_ch0'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

    plot_CvsIC_all1, ax_CvsIC_all1 = plt.subplots()
    for i in range(len(S1all_ic)):
        if S1all_caen_1[i] > 0: ax_CvsIC_all1.scatter(S1all_ic[i], S1all_caen_1[i], color=mapped_colors_all[i], s=25, label='Set 1')

    for i in range(len(S2all_ic)):
        if S2all_caen_1[i] > 0: ax_CvsIC_all1.scatter(S2all_ic[i], S2all_caen_1[i], color=mapped_colors_all[i], marker='^', s=25, label='Set 2')


    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_CvsIC_all1.colorbar(mappable, ax=ax_CvsIC_all1)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')

    ax_CvsIC_all1.set_xlabel('IC Charge (pC)')
    ax_CvsIC_all1.set_ylabel('CAEN Q (Channels)')
    ax_CvsIC_all1.set_xlim(-1, 20)
    ax_CvsIC_all1.set_ylim(600, 1600)

    # Makes the legend markers filled and outlined with black
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 1'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 2')
    ]
    ax_CvsIC_all1.legend(handles=legend_handles,loc='upper left')

    figgrid_name = 'April25_2026_CAENvsIC_ch1'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

    plot_CvsIC_all2, ax_CvsIC_all2 = plt.subplots()
    for i in range(len(S2all_ic)):
        if S1all_caen_2[i] > 0: ax_CvsIC_all2.scatter(S1all_ic[i], S1all_caen_2[i], color=mapped_colors_all[i], s=25, label='Set 1')

    for i in range(len(S2all_ic)):
        if S2all_caen_2[i] > 0: ax_CvsIC_all2.scatter(S2all_ic[i], S2all_caen_2[i], color=mapped_colors_all[i], marker='^', s=25, label='Set 2')


    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_CvsIC_all2.colorbar(mappable, ax=ax_CvsIC_all2)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')

    ax_CvsIC_all2.set_xlabel('IC Charge (pC)')
    ax_CvsIC_all2.set_ylabel('CAEN Q (Channels)')
    ax_CvsIC_all2.set_xlim(-1, 20)
    ax_CvsIC_all2.set_ylim(0, 500)

    # Makes the legend markers filled and outlined with black
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 1'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 2')
    ]
    ax_CvsIC_all2.legend(handles=legend_handles,loc='upper left')

    figgrid_name = 'April25_2026_CAENvsIC_ch2'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

    # plot ch 0 and 1
    plot_CvsIC_all01, ax_CvsIC_all01 = plt.subplots()
    for i in range(len(S1all_ic)):
        if S1all_caen_0[i] > 0: ax_CvsIC_all01.scatter(S1all_ic[i], S1all_caen_0[i], color=mapped_colors_all[i], s=25, label='Set 1')

    for i in range(len(S2all_ic)):
        if S2all_caen_0[i] > 0: ax_CvsIC_all01.scatter(S2all_ic[i], S2all_caen_0[i], color=mapped_colors_all[i], marker='^', s=25, label='Set 1')

    for i in range(len(S1all_ic)):
        if S1all_caen_1[i] > 0: ax_CvsIC_all01.scatter(S1all_ic[i], S1all_caen_1[i]*1.5, color=mapped_colors_all[i], s=25, label='Set 1')

    for i in range(len(S2all_ic)):
        if S2all_caen_1[i] > 0: ax_CvsIC_all01.scatter(S2all_ic[i], S2all_caen_1[i]*1.5, color=mapped_colors_all[i], marker='^', s=25, label='Set 2')

    mappable = cm.ScalarMappable(norm=norm_all, cmap=chosen_cmap)
    cbar = plot_CvsIC_all01.colorbar(mappable, ax=ax_CvsIC_all01)
    cbar.set_label(r'$\mathbf{Dump\ Width\ (\mu s)}$')

    ax_CvsIC_all01.set_xlabel('IC Charge (pC)')
    ax_CvsIC_all01.set_ylabel('CAEN Q (Channels)')
    ax_CvsIC_all01.set_xlim(-1, 20)
    ax_CvsIC_all01.set_ylim(600, 2100)

    # Makes the legend markers filled and outlined with black
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 1'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='black', markeredgecolor='black', markersize=8,
               ls='', label='Set 2')
    ]
    ax_CvsIC_all01.legend(handles=legend_handles,loc='upper left')

    figgrid_name = 'April25_2026_CAENvsIC_ch01'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

main()