########################################################
# Created on Feb 21 2026
# @author: Johnathan Phillips
# @email: j.s.phillips@wustl.edu

# Purpose: To unpack and analyze data from a medical proton beam on a scintillator

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


# Unpacks the CAEN data in csv format. Can choose to unpack all data or some using alldat
# Can also choose to import the waveforms (a lot of data) or just the first 6 columns
#   which contain the important information for each event
def Unpack(name, alldat, waves):
    if alldat is True and waves is True:
        data = np.genfromtxt(name, skip_header = True, delimiter = ';')
    elif alldat is False and waves is True:
        data = np.genfromtxt(name, skip_header = True, delimiter =';', max_rows = 50000)
    elif alldat is True and waves is False:
        data = np.genfromtxt(name, skip_header = True, delimiter =';', usecols = (0, 1, 2, 3, 4, 5))
    else:
        data = np.genfromtxt(name, skip_header = True, delimiter =';', max_rows = 10000, usecols = (0, 1, 2, 3, 4, 5))

    return data

# Function to discriminate on energy
def EDisc(energies,energies_short,times,channels,traces,elow,ehigh):

    energies_new = np.array([])
    energies_short_new = np.array([])
    times_new = np.array([])
    channels_new = np.array([])
    traces_new = np.ndarray((0, len(traces[0])))

    if (elow == 0):
        print("No lower energy cut")
        for i in range(len(energies)):
            if energies[i] < ehigh:
                energies_new = np.append(energies_new,energies[i])
                energies_short_new = np.append(energies_short_new,energies_short[i])
                times_new = np.append(times_new,times[i])
                channels_new = np.append(channels_new,channels[i])
                traces_new = np.append(traces_new,traces[i])

    elif (ehigh == 0):
        print("No higher energy cut")
        for i in range(len(energies)):
            if energies[i] > elow:
                energies_new = np.append(energies_new,energies[i])
                energies_short_new = np.append(energies_short_new,energies_short[i])
                times_new = np.append(times_new,times[i])
                channels_new = np.append(channels_new,channels[i])
                traces_new = np.append(traces_new,traces[i])
    else:
        print("Using lower and upper energy cuts")
        for i in range(len(energies)):
            if energies[i] < ehigh and energies[i] > elow:
                energies_new = np.append(energies_new,energies[i])
                energies_short_new = np.append(energies_short_new,energies_short[i])
                times_new = np.append(times_new,times[i])
                channels_new = np.append(channels_new,channels[i])
                traces_new = np.append(traces_new,[traces[i]],axis=0)

    return energies_new, energies_short_new, times_new, channels_new, traces_new

# Returns the data file, makes it easier to store them all
# Also determine filepath here
def GetFile():
    # file path, use 'Path' function to make compatible with windows/mac
    filepath = Path(r"/Users/jphillips409/Documents/SobotkaLab/ProtonBeam_ScintTests/Data/Feb14_2026")

    # Data files

    ###################################################

    # Feb 14, 2026
    # 10 pulses
    #file = r'ProtonBeam_Feb14_2026_10pulses_a/RAW/SDataR_ProtonBeam_Feb14_2026_10pulses_a.csv'
    #file = r'ProtonBeam_Feb14_2026_10pulses_b/RAW/SDataR_ProtonBeam_Feb14_2026_10pulses_b.csv'
    file = r'ProtonBeam_Feb14_2026_10pulses_c/RAW/SDataR_ProtonBeam_Feb14_2026_10pulses_c.csv'

    ###################################################

    return filepath / file

def main():
    # Read the CAEN data file in csv format
    # Different files available to process
    ###################################################

    # Remember to change your file path for your computer

    ###################################################
    # Constant filepath variable to get around the problem of backslashes in windows
    # The Path library will use forward slashes but convert them to correctly treat your OS
    # Also makes it easier to switch to a different computer

    datafile = GetFile()
    print("Analyzing file: ", datafile)

    alldat = True
    wavesdat = True
    data = Unpack(datafile, alldat, wavesdat)

    # Array to check event type
    chantype = []

    # The arrays used to store all the event information
    energy = []
    energy_short = []
    time = []
    channel = []
    traces = []

    satevents = 0 #  Counts the number of saturated events

    # Now fill in the data
    # Check event type before the loop, don't want to loop the event type check again and again
    for i in range(len(data)):
        channel.append(data[i][1])
        time.append(data[i][2])
        energy.append(data[i][3])
        if data[i][3] == 4095: satevents +=1
        energy_short.append(data[i][4])
        if wavesdat is True: traces.append(data[i][6:])

    print("Data unpacked")
    print("Num saturated events: ", satevents)

    # Convert the data into numpy arrays
    energy = np.array(energy)
    energy_short = np.array(energy_short)
    time = np.array(time)
    channel = np.array(channel)
    traces = np.array(traces)

    # Get energies discriminated on elow and ehigh
    # Will return all arrays using Ecuts
    elow = 50 # if 0, not used
    ehigh = 4000 # if 0, not used
    energy_disc, energy_short_disc, time_disc, channel_disc, traces_disc = EDisc(energy,energy_short, time, channel, traces,elow,ehigh)
    # Specify a number of bins
    nbins = 512

    # Plots the raw spectrum
    fig_rawE, ax_rawE = plt.subplots()
    ax_rawE.hist(energy, bins=nbins, range=[0, 4095])
    ax_rawE.set_title('Raw Energies')
    ax_rawE.set_xlabel('ADC Channel')
    ax_rawE.set_ylabel('Counts')
    plt.show()

    # Plot of energies vs index (raw)
    plot_EvsT, ax_EvsT = plt.subplots()
    xspace = np.linspace(0, len(channel),len(channel))
    ax_EvsT.scatter(xspace, energy)
    ax_EvsT.set_ylabel('ADC Channel')
    ax_EvsT.set_xlabel('index')
    plt.show()

    # Plot of energies vs Time (discriminated)
    plot_dEvsT, ax_dEvsT = plt.subplots()
    xspace = np.linspace(1, len(channel),len(channel))
    ax_dEvsT.scatter(time_disc/(10**12), energy_disc)
    ax_dEvsT.set_ylabel('ADC Channel')
    ax_dEvsT.set_xlabel('Time (s)')
    plt.show()

    # Plot traces
    xStart = 0
    xEnd = len(traces[1]) - 1
    wavetime = np.linspace(xStart, xEnd, len(traces[1]))
    # Convert sample number to time
    # CAEN samples every 2 ns
    wavetime = wavetime * 2
    plot_traces, ax_traces = plt.subplots()
    ax_traces.plot(wavetime,traces[4])
    #plt.xlim(500,1500)
    plt.show()

    # Plot discriminated traces 1 at a time
    plot_dtraces, ax_dtraces = plt.subplots()
    ax_dtraces.plot(wavetime,traces_disc[0])
    #plt.xlim(500,1500)
    plt.show()

    # make a window plot showing each real traces
    fig_multtrac, ax_multtrac = plt.subplots(5,5)
    row_num = 0
    col_num = 0
    for row in ax_multtrac:
        for col in row:
            if row_num + col_num == len(traces_disc): break
            col.plot(wavetime, traces_disc[row_num + col_num])
            col_num+=1

        if row_num + col_num == len(traces_disc): break
        row_num+=1

    plt.show()


main()
