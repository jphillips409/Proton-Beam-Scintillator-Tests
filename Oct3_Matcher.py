########################################################
# Created on June 2, 26
# @author: Johnathan Phillips
# @email: j.s.phillips@wustl.edu

# Purpose: Rematch data from Oct 3 2026
# Takes old matched events and exchanges with data from new built events

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

def Unpack(name):
    data = np.genfromtxt(name, skip_header = True, delimiter = '\t')
    return data

def Unpack_built(name):
    data = np.genfromtxt(name, skip_header = True)
    return data

def main():
    # Read in analyzed data in the form of txt files
    ###################################################

    # Remember to change your file path for your computer

    ###################################################
    # Constant filepath variable to get around the problem of backslashes in windows
    # The Path library will use forward slashes but convert them to correctly treat your OS
    # Also makes it easier to switch to a different computer

    # file path, use 'Path' function to make compatible with windows/mac
    filepath_m = Path(r"../Data/Oct3_2026/MatchedEvents")
    filepath_b = Path(r"../Data/Oct3_2026/BuiltEvents")

    matchdat = [] # for match files
    builtdat = [] # for built files

    runnum = 0
    matchdat.append([0]) # append a nothing run
    builtdat.append([0]) # append a nothing run
    # iterate over run numbers
    for i in range(1,35):
        # run 22 is bad
        if i == 22:
            matchdat.append([0])  # append a nothing run
            builtdat.append([0])  # append a nothing run
            continue
        runnum = i

        filename_m = r'run' + str(runnum) + r'_match.txt'
        file_m = filepath_m / filename_m
        matchdat.append(Unpack(file_m))

        filename_b = r'BuiltEvents_run' + str(runnum) + r'.txt'
        file_b = filepath_b / filename_b
        builtdat.append(Unpack_built(file_b))

    # Now look for events in matchdat that have no data (all zeros)
    # Use timing information based on known events to try and retrieve good events
    for i in range(1, 35):
        if i == 22: continue

        firstevent = -1

        # loop over the rows
        # find first event that has good data
        for j in range(0, len(matchdat[i])):
            if (matchdat[i][j][4] != 0):
                firstevent = j
            elif (matchdat[i][j][5] != 0):
                firstevent = j
            elif (matchdat[i][j][6] != 0):
                firstevent = j
            else: continue

            if firstevent != -1: break

        # if never found, go to next run
        if firstevent == -1: continue

        # get time stamp of first event
        firsttstamp = matchdat[i][j][3]

        # find index of first associated built event using energies (CAEN Q)
        firstmatch_b = -1
        for j in range(0,len(builtdat[i])):
            if matchdat[i][firstevent][4] == int(builtdat[i][j][1]) and builtdat[i][j][1] != 0:
                firstmatch_b = j
            elif matchdat[i][firstevent][5] == int(builtdat[i][j][3]) and builtdat[i][j][3] != 0:
                firstmatch_b = j
            elif matchdat[i][firstevent][6] == int(builtdat[i][j][5]) and builtdat[i][j][5] != 0:
                firstmatch_b = j
            else: continue

            if firstmatch_b != -1: break

        # if not firsttstamp_b, go to next run
        if firstmatch_b == -1: continue

        # get shift needed
        tshift = firsttstamp - builtdat[i][firstmatch_b][0]

        # now shift tstamps of entire built event to line up firstmatch_b and firsttstamp
        # First recalc tstamp using reference point
        built_newt = [] # to hold new t stamps
        for j in range(0,len(builtdat[i])):
            built_newt.append(builtdat[i][j][0] + tshift)

        # now go through and use np.isclose to fill gaps
        # 5% is close threshold, should be good enough
        jmove = 0
        for j in range(0,len(matchdat[i])):
            # skip j if already known
            if matchdat[i][j][4] != 0 or matchdat[i][j][5] != 0 or matchdat[i][j][6] != 0: continue
            if j >= len(builtdat[i]): break
            for n in range(0+j,len(builtdat[i])):
                # must be completely empty
                if matchdat[i][j][4] == 0 and matchdat[i][j][5] == 0 and matchdat[i][j][6] == 0:
                    if np.isclose(matchdat[i][j][3],built_newt[n],atol=0.3,rtol=0):
                        matchdat[i][j][4] = builtdat[i][n][1]
                        matchdat[i][j][5] = builtdat[i][n][3]
                        matchdat[i][j][6] = builtdat[i][n][5]
                        j = n+1
                        break


# get target indices
    targetind = [5,6,7]
    # build the events in proper order by matching built events and matched events
    for i in range(1,35):
        if i == 22: continue
        # get length of matched event
        num_rows = len(matchdat[i])
        col0 = []
        col2 = []
        col4 = []
        jmove = 0
        #print(i)
        for j in range(0,num_rows):
            # compare E0, easy solution if it exists
            # should be time ordered
            if jmove >= len(builtdat[i]):
                col0.append(0)
                col2.append(0)
                col4.append(0)
                continue
            for n in range(jmove,len(builtdat[i])):
                # check 0
                if matchdat[i][j][4] == int(builtdat[i][n][1]) and builtdat[i][n][1] != 0:
                    col0.append(builtdat[i][n][2])
                    col2.append(builtdat[i][n][4])
                    col4.append(builtdat[i][n][6])
                    matchdat[i][j][4] = builtdat[i][n][1]
                    matchdat[i][j][5] = builtdat[i][n][3]
                    matchdat[i][j][6] = builtdat[i][n][5]
                    jmove = n+1
                    break
                # check 2
                elif matchdat[i][j][5] == int(builtdat[i][n][3]) and builtdat[i][n][3] != 0:
                    col0.append(builtdat[i][n][2])
                    col2.append(builtdat[i][n][4])
                    col4.append(builtdat[i][n][6])
                    matchdat[i][j][4] = builtdat[i][n][1]
                    matchdat[i][j][5] = builtdat[i][n][3]
                    matchdat[i][j][6] = builtdat[i][n][5]
                    jmove = n+1
                    break
                #check 4
                elif matchdat[i][j][6] == int(builtdat[i][n][5]) and builtdat[i][n][5] != 0:
                    col0.append(builtdat[i][n][2])
                    col2.append(builtdat[i][n][4])
                    col4.append(builtdat[i][n][6])
                    matchdat[i][j][4] = builtdat[i][n][1]
                    matchdat[i][j][5] = builtdat[i][n][3]
                    matchdat[i][j][6] = builtdat[i][n][5]
                    jmove = n+1
                    break
                elif matchdat[i][j][4] == 0 and matchdat[i][j][5] == 0 and matchdat[i][j][6] == 0:
                    col0.append(0)
                    col2.append(0)
                    col4.append(0)
                    break
                elif n == len(builtdat[i]) - 1: # just to catch stuff
                    col0.append(0)
                    col2.append(0)
                    col4.append(0)
                    break
                #print(i, ' ', j, ' ',n)
        newcols = np.column_stack((col0,col2,col4))
        # insert columns into matchdat
        matchdat[i] = np.insert(matchdat[i],targetind,newcols,axis=1)

    # open output file
    ofilepath = Path(r"../Data/Oct3_2026/MatchedEvents")
    #print(matchdat[1][0][0])
    # now go through and match data based on energies
    # time stamps are fickle, but energies are not
    # Indices of energies need to be tweaked based on what has been added to built events
    for i in range(1, 35):
        if i == 22: continue
        ofilen = r'run' + str(i) + r'_match_live.txt'
        ofile = ofilepath / ofilen
        with open(ofile, 'w') as f:
            print("Pulse\tCyc T stamp (us)\tDoseplane_pC\tCAEN rel tstamp (ms)\tch 0 Q (Chan)\tch 0 new Q\tch 2 Q (Chan)\tch 2 new Q\tch 4 Q (Chan)\tch 4 new Q",file=f)
            # print matche data
            for row in matchdat[i]:
                line = "\t".join(map(str, row))
                #if i == 1: print(line)
                f.write(line + "\n")

main()