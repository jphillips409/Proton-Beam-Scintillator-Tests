########################################################
# Created on June 2 2026
# @author: Johnathan Phillips
# @email: j.s.phillips@wustl.edu

# Purpose: To unpack and analyze data from a medical proton beam on a scintillator
# This is a modified version of Unpack.py, use this for multi-channel time matching

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
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

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

# 1. Create a clean base layout template
mpl_style = go.layout.Template()

# 2. Replicate Matplotlib's layout details
mpl_style.layout = go.Layout(
    template="plotly_white",        # Uses a clean white background canvas
    title=dict(
        font=dict(size=16, family="DejaVu Sans, Arial"),
        x=0.5,                      # Center-align the title like Matplotlib
        xanchor="center"
    ),
    font=dict(
        family="DejaVu Sans, Arial", # Use default Matplotlib fonts
        size=12,
        color="black"
    ),
    # X-Axis Styling
    xaxis=dict(
        showline=True,              # Explicit outer frame border line
        linewidth=1.2,
        linecolor="black",
        mirror=True,                # Mirrors the border on the top of the plot
        ticks="inside",             # Inward-facing tick marks
        tickwidth=1.2,
        tickcolor="black",
        showgrid=True,              # Optional: set to False if you want clean backgrounds
        gridcolor="rgba(220, 220, 220, 0.5)",
    ),
    # Y-Axis Styling
    yaxis=dict(
        showline=True,              # Explicit outer frame border line
        linewidth=1.2,
        linecolor="black",
        mirror=True,                # Mirrors the border on the right of the plot
        ticks="inside",             # Inward-facing tick marks
        tickwidth=1.2,
        tickcolor="black",
        showgrid=True,
        gridcolor="rgba(220, 220, 220, 0.5)",
    ),
    # Global default behavior for specific charts
    bargap=0                        # Eliminates spaces between histogram/bar bins globally
)

# 3. Register and set as your absolute global environment default
pio.templates["matplotlib_clean"] = mpl_style
pio.templates.default = "matplotlib_clean"

figurepath = Path('OutputFigures') # Go outside of directory and enter Paper_Figures
figurepath = '..' / figurepath

# Unpacks the CAEN data in csv format. Can choose to unpack all data or some using alldat
# Can also choose to import the waveforms (a lot of data) or just the first 6 columns
#   which contain the important information for each event
# Flag parameter (index 5) must be converted from Hex to 32int
def Unpack(name, alldat, waves):
    if alldat is True and waves is True:
        data = np.genfromtxt(name, skip_header = True, delimiter = ';',converters={5: lambda s: float.fromhex(s)})
    elif alldat is False and waves is True:
        data = np.genfromtxt(name, skip_header = True, delimiter =';', max_rows = 50000,converters={5: lambda s: float.fromhex(s)})
    elif alldat is True and waves is False:
        data = np.genfromtxt(name, skip_header = True, delimiter =';', usecols = (0, 1, 2, 3, 4, 5),converters={5: lambda s: float.fromhex(s)})
    else:
        data = np.genfromtxt(name, skip_header = True, delimiter =';', max_rows = 10000, usecols = (0, 1, 2, 3, 4, 5),converters={5: lambda s: float.fromhex(s)})

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

# needed for plotly nonsense if you want to use
# makes the binning similar to root for the interactive plot
def plot_1D_spectrum(energy_data, title="Raw Spectrum", nbins=512, x_min=0, x_max=4095):
    """
    Computes a fixed-bin histogram and plots it in Plotly with a
    strict classic Matplotlib layout containing primary and secondary ticks.
    """
    # 1. Compute static histogram via NumPy
    counts, bin_edges = np.histogram(energy_data, bins=nbins, range=[x_min, x_max])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2. Render base chart
    fig = px.bar(
        x=bin_centers,
        y=counts,
        title=title,
        labels={"x": "Q (ADC Channels)", "y": "Counts"}
    )

    # 3. Explicitly force Matplotlib styling directly onto this figure
    fig.update_layout(
        template="plotly_white",  # Clean white canvas
        bargap=0,  # Remove all space between bins
        title=dict(
            x=0.5,  # Center-align the title
            xanchor="center",
            font=dict(size=16, family="DejaVu Sans, Arial", color="black")
        ),
        font=dict(family="DejaVu Sans, Arial", size=12, color="black"),
        margin=dict(t=60, b=60, l=60, r=40),
    )

    # Clean up the bars to look like a solid continuous spectrum
    fig.update_traces(marker_line_width=0)

    # Configure X-Axis: Box border, inside ticks, and specific range
    fig.update_xaxes(
        range=[x_min, x_max],
        showline=True,
        linewidth=1.5,
        linecolor="black",
        mirror="allticks",  # Crucial: Mirrors both primary AND secondary ticks to the top border
        ticks="inside",  # Forces major ticks inside the plot
        tickwidth=1.5,
        ticklen=6,  # Longer length for major ticks
        tickcolor="black",
        showgrid=False,
        # Secondary / Minor Ticks Configuration
        minor=dict(
            ticks="inside",  # Forces minor ticks inside the plot
            tickwidth=1.0,
            ticklen=3.5,  # Shorter length for minor ticks
            tickcolor="black",
            tickmode="auto"  # Automatically places clean subdivisions between major intervals
        )
    )

    # Configure Y-Axis: Box border and inside ticks
    fig.update_yaxes(
        showline=True,
        linewidth=1.5,
        linecolor="black",
        mirror="allticks",  # Crucial: Mirrors both primary AND secondary ticks to the right border
        ticks="inside",  # Forces major ticks inside the plot
        tickwidth=1.5,
        ticklen=6,  # Longer length for major ticks
        tickcolor="black",
        showgrid=False,
        # Secondary / Minor Ticks Configuration
        minor=dict(
            ticks="inside",  # Forces minor ticks inside the plot
            tickwidth=1.0,
            ticklen=3.5,  # Shorter length for minor ticks
            tickcolor="black",
            tickmode="auto"
        )
    )

    return fig

# Returns the data file, makes it easier to store them all
# Also determine filepath here
def GetFile():
    # file path, use 'Path' function to make compatible with windows/mac
    filepath = Path(r"../Data/Feb14_2026")

    # Data files

    ###################################################

    # Feb 14, 2026
    # 10 pulses
    #file = r'ProtonBeam_Feb14_2026_10pulses_a/RAW/SDataR_ProtonBeam_Feb14_2026_10pulses_a.csv'
    #file = r'ProtonBeam_Feb14_2026_10pulses_b/RAW/SDataR_ProtonBeam_Feb14_2026_10pulses_b.csv'
    #file = r'ProtonBeam_Feb14_2026_10pulses_c/RAW/SDataR_ProtonBeam_Feb14_2026_10pulses_c.csv'

    # 20 pulses
    #file = r'ProtonBeam_Feb14_2026_20pulses_a/RAW/SDataR_ProtonBeam_Feb14_2026_20pulses_a.csv'
    #file = r'ProtonBeam_Feb14_2026_20pulses_b/RAW/SDataR_ProtonBeam_Feb14_2026_20pulses_b.csv'
    #file = r'ProtonBeam_Feb14_2026_20pulses_c/RAW/SDataR_ProtonBeam_Feb14_2026_20pulses_c.csv'

    # 100 pulses
    #file = r'ProtonBeam_Feb14_2026_100pulses_a/RAW/SDataR_ProtonBeam_Feb14_2026_100pulses_a.csv'
    #file = r'ProtonBeam_Feb14_2026_100pulses_b/RAW/SDataR_ProtonBeam_Feb14_2026_100pulses_b.csv'
    #file = r'ProtonBeam_Feb14_2026_100pulses_c/RAW/SDataR_ProtonBeam_Feb14_2026_100pulses_c.csv'

    # 10 pulses - no scintillator
    #file = r'ProtonBeam_Feb14_2026_BareFib_a/RAW/SDataR_ProtonBeam_Feb14_2026_BareFib_a.csv'
    #file = r'ProtonBeam_Feb14_2026_BareFib_b/RAW/SDataR_ProtonBeam_Feb14_2026_BareFib_b.csv'
    #file = r'ProtonBeam_Feb14_2026_BareFib_c/RAW/SDataR_ProtonBeam_Feb14_2026_BareFib_c.csv'

    ###################################################

    # Apr 25, 2026
    filepath = Path(r"../Data/April25_2026")


    # 50 pulses

    # 15.15 us width
    #file = r'ProtonBeam_April25_2026_Run1/RAW/SDataR_ProtonBeam_April25_2026_Run1.csv'
    #file = r'ProtonBeam_April25_2026_Run2/RAW/SDataR_ProtonBeam_April25_2026_Run2.csv'

    # 5 us width
    #file = r'ProtonBeam_April25_2026_Run3/RAW/SDataR_ProtonBeam_April25_2026_Run3.csv'
    #file = r'ProtonBeam_April25_2026_Run4/RAW/SDataR_ProtonBeam_April25_2026_Run4.csv'

    # 10 us width
    #file = r'ProtonBeam_April25_2026_Run5/RAW/SDataR_ProtonBeam_April25_2026_Run5.csv'
    #file = r'ProtonBeam_April25_2026_Run6/RAW/SDataR_ProtonBeam_April25_2026_Run6.csv'

    # 17 us width
    #file = r'ProtonBeam_April25_2026_Run7/RAW/SDataR_ProtonBeam_April25_2026_Run7.csv'
    #file = r'ProtonBeam_April25_2026_Run8/RAW/SDataR_ProtonBeam_April25_2026_Run8.csv'

    # 24 us width
    #file = r'ProtonBeam_April25_2026_Run9/RAW/SDataR_ProtonBeam_April25_2026_Run9.csv'
    #file = r'ProtonBeam_April25_2026_Run10/RAW/SDataR_ProtonBeam_April25_2026_Run10.csv'

    # 30 us width
    #file = r'ProtonBeam_April25_2026_Run11/RAW/SDataR_ProtonBeam_April25_2026_Run11.csv'
    #file = r'ProtonBeam_April25_2026_Run12/RAW/SDataR_ProtonBeam_April25_2026_Run12.csv'

    # 2.3 us width
    #file = r'ProtonBeam_April25_2026_Run13/RAW/SDataR_ProtonBeam_April25_2026_Run13.csv'
    #file = r'ProtonBeam_April25_2026_Run14/RAW/SDataR_ProtonBeam_April25_2026_Run14.csv'

    # 10 us width with beam rate divider of 2
    #file = r'ProtonBeam_April25_2026_Run15/RAW/SDataR_ProtonBeam_April25_2026_Run15.csv'


    ###################################################

    # Oct 3, 2026
    # 3 Channels - 0, 2, 4 (diff. memory banks)
    # channel 0 - yellow, first block, side scintillator
    # channel 2 - orange, first block, top scintillator
    # channel 4 - brown/red, second block, side optical
    # Types of runs:
    # 1 - 9: 750 Hz, 25 us PW, #P = 10, 50, 100 (3 each)
    # 10 - 12: #P = 100, 375 Hz, 25 us
    # 13 - 15: #P = 100, 250 Hz, 25 us
    # 16 - 18: #P = 100, 750 Hz, 20 us
    # 19 - 21: #P = 100, 750 Hz, 15 us
    # 22 BAD
    # 23 - 25: #P = 100, 750 Hz, 10 us
    # 26 - 28: #P = 100, 750 Hz, 22 us
    # 29 - 31: #P = 100, 750 Hz, 17 us
    # 32 - 34: #P = 100, 750 Hz, 19 us

    filepath = Path(r"../Data/Oct3_2026")

    # runs
    file = r'pbeam_oct3_run1/RAW/SDataR_pbeam_oct3_run1.csv'
    #file = r'pbeam_oct3_run2/RAW/SDataR_pbeam_oct3_run2.csv'
    #file = r'pbeam_oct3_run3/RAW/SDataR_pbeam_oct3_run3.csv'

    #file = r'pbeam_oct3_run4/RAW/SDataR_pbeam_oct3_run4.csv'
    #file = r'pbeam_oct3_run5/RAW/SDataR_pbeam_oct3_run5.csv'
    #file = r'pbeam_oct3_run6/RAW/SDataR_pbeam_oct3_run6.csv'

    #file = r'pbeam_oct3_run7/RAW/SDataR_pbeam_oct3_run7.csv'
    #file = r'pbeam_oct3_run8/RAW/SDataR_pbeam_oct3_run8.csv'
    #file = r'pbeam_oct3_run9/RAW/SDataR_pbeam_oct3_run9.csv'

    #file = r'pbeam_oct3_run10/RAW/SDataR_pbeam_oct3_run10.csv'
    #file = r'pbeam_oct3_run11/RAW/SDataR_pbeam_oct3_run11.csv'
    #file = r'pbeam_oct3_run12/RAW/SDataR_pbeam_oct3_run12.csv'

    #file = r'pbeam_oct3_run13/RAW/SDataR_pbeam_oct3_run13.csv'
    #file = r'pbeam_oct3_run14/RAW/SDataR_pbeam_oct3_run14.csv'
    #file = r'pbeam_oct3_run15/RAW/SDataR_pbeam_oct3_run15.csv'

    #file = r'pbeam_oct3_run16/RAW/SDataR_pbeam_oct3_run16.csv'
    #file = r'pbeam_oct3_run17/RAW/SDataR_pbeam_oct3_run17.csv'
    #file = r'pbeam_oct3_run18/RAW/SDataR_pbeam_oct3_run18.csv'

    #file = r'pbeam_oct3_run19/RAW/SDataR_pbeam_oct3_run19.csv'
    #file = r'pbeam_oct3_run20/RAW/SDataR_pbeam_oct3_run20.csv'
    #file = r'pbeam_oct3_run21/RAW/SDataR_pbeam_oct3_run21.csv'

    #file = r'pbeam_oct3_run23/RAW/SDataR_pbeam_oct3_run23.csv'
    #file = r'pbeam_oct3_run24/RAW/SDataR_pbeam_oct3_run24.csv'
    #file = r'pbeam_oct3_run25/RAW/SDataR_pbeam_oct3_run25.csv'

    #file = r'pbeam_oct3_run26/RAW/SDataR_pbeam_oct3_run26.csv'
    #file = r'pbeam_oct3_run27/RAW/SDataR_pbeam_oct3_run27.csv'
    #file = r'pbeam_oct3_run28/RAW/SDataR_pbeam_oct3_run28.csv'

    #file = r'pbeam_oct3_run29/RAW/SDataR_pbeam_oct3_run29.csv'
    #file = r'pbeam_oct3_run30/RAW/SDataR_pbeam_oct3_run30.csv'
    #file = r'pbeam_oct3_run31/RAW/SDataR_pbeam_oct3_run31.csv'

    #file = r'pbeam_oct3_run32/RAW/SDataR_pbeam_oct3_run32.csv'
    #file = r'pbeam_oct3_run33/RAW/SDataR_pbeam_oct3_run33.csv'
    #file = r'pbeam_oct3_run34/RAW/SDataR_pbeam_oct3_run34.csv'


    # Digitizer testing data
    #filepath = Path(r"../Data/Testing")

    #file = r'SDataR_PulserTest_50waves_June23_26.csv'

    # Run log excel contains info
    #file = r'SDataR_PulserTest_June23_26_run1.csv'
    #file = r'SDataR_PulserTest_June23_26_run2.csv'
    #file = r'SDataR_PulserTest_June23_26_run3.csv'
    #file = r'SDataR_PulserTest_June23_26_run4.csv'
    #file = r'SDataR_PulserTest_June23_26_run5.csv'
    #file = r'SDataR_PulserTest_June23_26_run6.csv'
    #file = r'SDataR_PulserTest_June23_26_run7.csv'
    #file = r'SDataR_PulserTest_June23_26_run8.csv'
    #file = r'SDataR_PulserTest_June23_26_run9.csv'
    #file = r'SDataR_PulserTest_June23_26_run10.csv'
    #file = r'SDataR_PulserTest_June23_26_run11.csv'
    #file = r'SDataR_PulserTest_June23_26_run12.csv'
    #file = r'SDataR_PulserTest_June23_26_run13.csv'
    #file = r'SDataR_PulserTest_June23_26_run14.csv'
    #file = r'SDataR_PulserTest_June23_26_run15.csv'



    return filepath / file, file

def main():
    # Read the CAEN data file in csv format
    # Different files available to process
    ###################################################

    # Remember to change your file path for your computer

    ###################################################
    # Constant filepath variable to get around the problem of backslashes in windows
    # The Path library will use forward slashes but convert them to correctly treat your OS
    # Also makes it easier to switch to a different computer

    datafile, filecsv = GetFile()
    print("Analyzing file: ", datafile)

    # Reads the data file string and gets some of the name
    # Useful for saving figures
    data_string = list(filecsv)
    savename = []
    for i in range(len(data_string)):
        # First search for RAW
        if i != 0 and data_string[i] == 'W' and data_string[i-1] == 'A' and data_string[i-2] == 'R':
            for j in range(i + 1, len(data_string)):
                if data_string[j] == '_' and data_string[j-1] == 'm' and data_string[j-2] == 'a' and data_string[j-3] == 'e':
                    for k in range(j + 1, len(data_string)):
                        if data_string[k] == '.': break
                        savename.append(data_string[k])
                else: continue
        else: continue
    savename = ''.join(savename)
    print('Figure save name: ', savename)

    alldat = True
    wavesdat = False
    # Turn wavesgrid on if you want to plot a large grid of E discriminated waves
    wavesgrid = False
    data = Unpack(datafile, alldat, wavesdat)

    # Array to check event type
    chantype = []

    # The arrays used to store all the event information
    energy = []
    energy_short = []
    time = []
    channel = []
    traces = []
    flags = []
    probe=[]

    satevents = 0 #  Counts the number of saturated events

    # Now fill in the data
    # Check event type before the loop, don't want to loop the event type check again and again
    for i in range(len(data)):
        probeoff = 0
        #if data[i][1] != 0: continue
        channel.append(data[i][1])
        time.append(data[i][2])
        energy.append(data[i][3])
        if data[i][5] == 16512: satevents +=1
        energy_short.append(data[i][4])
        flags.append(data[i][5])
        if data[i][6] == 1: # checks if probe value is there, not sure what it does
            probe.append(data[i][6])
            probeoff = 1
        if wavesdat is True: traces.append(data[i][6 + probeoff:])

    print("Data unpacked")
    print("Num saturated events: ", satevents)

    # Convert the data into numpy arrays
    energy = np.array(energy)
    energy_short = np.array(energy_short)
    time = np.array(time)
    channel = np.array(channel)
    traces = np.array(traces)
    flags = np.array(flags)

  #  for i in range(len(flags)):
       # if flags[i] != 16384:
            #print('Flags other than 0x4000 at index: ', i, ' and flag: ', f'0x{int(flags[i]):x}')

    # Time match channels
    # Match them to ch 0 or 1
    # For the April 25 runs, 0 and 1 are exclusive of each other, some CAEN memory issue
    ch02_e = [] # pairs relative to 0
    ch02_t = []
    ch12_e = [] # pairs relative to 1
    ch12_t = []

    # Loop over and look for neighboring channels with close times
    # 1.32 ms dump spacing, so pretty wide window
    twindow = 10000000 # 10 us window I believe
    i = 0
    while i < len(channel) - 1:
        # Make sure to get out if at end
        if i == len(channel) - 1: break
        if energy[i] < 10:
            i += 1
            continue
        match = False
        match02 = False # individual matching
        match12 = False

        # for modifying i
        incr = 0

        # 3 channels, so only look 2 ahead
        # This can stay as i + 3
        for j in range(i+1,len(channel)):
            # break if at end
            if j == len(channel): break

            # some threshold needed
            if (energy[j] < 10): continue
            # if next channel is the same, continue
            # have to do entire loop now due to weirdness with datastream
            if channel[i] == channel[j]: continue
            if channel[i] == 0 and channel[j] == 1: continue
            if channel[i] == 1 and channel[j] == 0: continue
            # Check if within coarse t window
            if abs(time[i] - time[j]) <= twindow:
                match = True

                # get index for 0
                zi = i
                oi = j # other index
                if channel[j] == 0 or channel[j] == 1:
                    zi = j
                    oi = i

                # gets a little complicated here. Consider pairs and triplets
                if channel[zi] == 0 and channel[oi] == 2:
                    match02 = True
                    ch02_e.append([energy[zi],energy[oi]])
                    ch02_t.append([time[zi],time[oi]])
                    incr += 1


                if channel[zi] == 1 and channel[oi] == 2:
                    match12 = True
                    ch12_e.append([energy[zi], energy[oi]])
                    ch12_t.append([time[zi], time[oi]])
                    incr += 1
                else:
                    incr += 1
                    break
            else: continue

            if match == True: break


        # if matched, do not increment again
        if match == True:
            i += 1
            continue
        i += 1

    print(ch02_e)

    print(ch12_e)

    # Get energies discriminated on elow and ehigh
    # Will return all arrays using Ecuts
    elow = 50 # if 0, not used
    ehigh = 4000 # if 0, not used
    energy_disc, energy_short_disc, time_disc, channel_disc, traces_disc = EDisc(energy,energy_short, time, channel, traces,elow,ehigh)

    # Array of times relative to something
    # Choose first pulse as t = 0
    time_disc_rel = time_disc - time_disc[0]
    print("Time array relative to 1st pulse: ")
    #for i in range(len(time_disc_rel)):
     #print(time_disc_rel[i]/(10**9), " ", energy_disc[i]/100)

    #print("Time array relative to 1st pulse: ", time_disc_rel/(10**12))

    # Specify a number of bins
    nbins = 512

    bin_size = (4095 - 0 + 1) / 512  # Equals 8 channels per bin

    #print('Number of energies above/below thresholds: ', len(energy_disc))

    # Plots the raw spectrum
    fig_rawE, ax_rawE = plt.subplots()
    ax_rawE.hist(energy, bins=nbins, range=[0, 4095])
    ax_rawE.set_title('Raw Energies')
    ax_rawE.set_xlabel('Q (ADC Channels)')
    ax_rawE.set_ylabel('Counts')
    plt.show()

    #plot raw spectrum using plotly
    fig_pl = plot_1D_spectrum(energy,"plotly raw spectrum",nbins,0,4095)
    fig_pl.show()  # Automatically opens an interactive plot in your default web browser

    #for i in range(len(channel)):
        #if flags[i] == 16512: print('Saturated event index: ', i)

    # Plot of energies vs index (raw)
    plot_EvsT, ax_EvsT = plt.subplots()
    xspace = np.linspace(0, len(channel)-1,len(channel))
    ax_EvsT.scatter(xspace, energy)
    ax_EvsT.set_ylabel('Q (ADC Channels)')
    ax_EvsT.set_xlabel('Event Index')
    figgrid_name = savename + '_EvsID'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()
    plt.show()

    # Plot of energies vs Time (raw)
    plot_EvsT, ax_EvsT = plt.subplots()
    ax_EvsT.scatter(time/(10**12), energy)
    ax_EvsT.set_ylabel('Q (ADC Channels)')
    ax_EvsT.set_xlabel('Time since Run Start (s)')
    #ax_EvsT.set_ylim(900,1500)
    #ax_EvsT.set_ylim(0,600) # Without scintillator
    plt.show()

    # Plot of energies vs Time (discriminated)
    plot_dEvsT, ax_dEvsT = plt.subplots()
    ax_dEvsT.scatter(time_disc_rel/(10**9), energy_disc)
    ax_dEvsT.set_ylabel('Q (ADC Channels)')
    ax_dEvsT.set_xlabel('Time Since First Pulse (ms)')
   # ax_dEvsT.set_xlim(2.54,None)
    #ax_dEvsT.set_ylim(0,600) # Without scintillator

    figgrid_name = savename + '_discEvsT'
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()

    # Debugging print statements
    count0 = 0
    count1 = 0
    count2 = 0
    i=0
    plot_1vsT, ax_1vsT = plt.subplots()
    for i in range(len(channel)):
        if energy[i] > 50 and energy[i] < 2000 and energy[i] != 4095:
            #print(channel[i], ' ', time[i])
            if (channel[i] == 0):
                if count0 == 0: ax_1vsT.scatter(time[i] / (10 ** 9), 0.6,color='blue',label='Ch 0 (Scint 1)')
                else: ax_1vsT.scatter(time[i] / (10 ** 9), 0.6,color='blue')
                count0 += 1
            if (channel[i] == 1 or channel[i] == 4):
                if count1 == 0: ax_1vsT.scatter(time[i] / (10 ** 9), 0.5, color='red',label='Ch 1 (Scint 2 w/ Resistor)')
                else: ax_1vsT.scatter(time[i] / (10 ** 9), 0.5,color='red')
                count1 += 1
            if (channel[i] == 2):
                if count2 == 0: ax_1vsT.scatter(time[i] / (10 ** 9), 0.4, color='green',label='Ch 2 (Optical)')
                else: ax_1vsT.scatter(time[i] / (10 ** 9), 0.4,color='green')
                count2 += 1

    ax_1vsT.set_ylim(0,1)
    ax_1vsT.set_xlabel('Timestamp (ms)')
    ax_1vsT.set_ylabel('Arbitrary Units')
    plt.legend()
    plt.show()

    print('count 0: ', count0, ' count 1 ', count1, ' sum ', count0 + count1)

    # first, find the number of dumps for each channel (discriminated)
    N0 = 0
    N1 = 0
    N2 = 0
    for i in range(len(channel)):
        if (energy[i] > 25 and energy[i] < 3000):
            if channel[i] == 0: N0 += 1
            if channel[i] == 1: N1 += 1
            if channel[i] == 2: N2 += 1

    # complicated print statement
    # want to print out channels 0, 1, and 2 in a way that is easy for excel to handle
    # loop over 0 and 1, and look for the pairs 02 and 12
    # for 02 and 12 events, 0 and 1 have T stamp priority
    #####
    # print like this
    # Dump T stamp      ch 0 E      ch 1 E      ch 2 E
    #####
    counter0 = 0
    counter1 = 0
    counter2 = 0
    # already time ordered, so FOR will work
    # want to print time difference relative to first one
    TS1 = -1
    print('Large discriminated event dumper')
    print('Tstamp (ms)   ch 0 E     ch 1 E     ch 2 E')
    for i in range(len(channel)):
        # find earliest time for each event (discriminated)
        if (energy[i] < 25 or energy[i] > 3000): continue

        # will never have 0 - 1 pair, might have 2
        if channel[i] == 0:
            if energy[i] < 100: continue
            if TS1 == -1: TS1 = i
            counter0 += 1
            ch2E = 0
            rtime = time[i]
            if TS1 != -1: rtime = time[i] - time[TS1]
            for j in range(len(ch02_e)):
                if time[i] == ch02_t[j][0]: ch2E = ch02_e[j][1]
            print(rtime/10**9, ' ', energy[i], ' 0 ', ch2E)

        if channel[i] == 1:
            if energy[i] < 100: continue
            if TS1 == -1: TS1 = i
            counter1 += 1
            ch2E = 0
            rtime = time[i]
            if TS1 != -1: rtime = time[i] - time[TS1]
            for j in range(len(ch12_e)):
                if time[i] == ch12_t[j][0]: ch2E = ch12_e[j][1]
            print(rtime/10**9, ' 0 ', energy[i], ' ', ch2E)

        # for loner channel 2s
        # make sure they have not been matched
        if channel[i] == 2:
            if TS1 == -1: TS1 = i
            counter2 += 1
            tmatch = False
            rtime = time[i]
            if TS1 != -1: rtime = time[i] - time[TS1]
            for j in range(len(ch12_e)):
                if time[i] == ch12_t[j][1]: tmatch = True
            for j in range(len(ch02_e)):
                if time[i] == ch02_t[j][1]: tmatch = True
            if tmatch == False: print(rtime/10**9, ' 0 0 ', energy[i])




    # Weird function time
    # Want to iterate over all (good) wave forms and average their values as a function of time.
    # Should set a threshold to remove outlier (bad integral, shifted in t) wave forms
    # Want to produce an averaged wave form

    # Let's start with ch 0
    wavecount0 = 0 # use this for averaging
    avgwave0 = np.zeros(len(traces[0])) # grab trace array shape
    for i in range(len(channel)):
        # only interested in channel 0 rn
        if channel[i] != 0: continue
        if energy[i] < 1000 or energy[i] == 4095: continue
        wavecount0 += 1
        # loop over trace and add to avgwave0
        for j in range(len(traces[i])):
            avgwave0[j] += traces[i][j]

    # now average the waveform
    for i in range(len(avgwave0)):
        avgwave0[i] = avgwave0[i]/wavecount0

    # Now subtract the baseline and flip, take the first 100 points
    avgwave0_bkg = np.average(avgwave0[0:99])
    for i in range(len(avgwave0)):
        avgwave0[i] = avgwave0[i] - avgwave0_bkg
        avgwave0[i] = -avgwave0[i]

    # Plot ch 0 avg trace
    xStart = 0
    xEnd = len(traces[0]) - 1
    wavetime = np.linspace(xStart, xEnd, len(traces[0]))
    # Convert sample number to time
    # CAEN samples every 2 ns
    wavetime = wavetime * 2

    # Plot avg trace traces 1 at a time
    plot_avgtrace0, ax_avgtrace0 = plt.subplots()
    ax_avgtrace0.plot(wavetime,avgwave0)
    ax_avgtrace0.set_xlabel('Time (ns)')
    ax_avgtrace0.set_ylabel('ADC Channels')
    #ax_dtraces.set_ylim(12500,13200)
    #plt.xlim(701,1299)
    plt.show()

    # Now ch 1
    wavecount1 = 0  # use this for averaging
    avgwave1 = np.zeros(len(traces[0]))  # grab trace array shape
    for i in range(len(channel)):
        # only interested in channel 0 rn
        if channel[i] != 1 or energy[i] < 500 or energy[i] == 4095: continue
        wavecount1 += 1
        # loop over trace and add to avgwave0
        for j in range(len(traces[i])):
            avgwave1[j] += traces[i][j]

    # now average the waveform
    for i in range(len(avgwave1)):
        avgwave1[i] = avgwave1[i] / wavecount1

    # Now subtract the baseline and flip, take the first 100 points
    avgwave1_bkg = np.average(avgwave1[0:99])
    for i in range(len(avgwave1)):
        avgwave1[i] = avgwave1[i] - avgwave1_bkg
        avgwave1[i] = -avgwave1[i]

    # Plot ch 1 avg trace
    # Plot avg trace traces 1 at a time
    plot_avgtrace1, ax_avgtrace1 = plt.subplots()
    ax_avgtrace1.plot(wavetime, avgwave1)
    ax_avgtrace1.set_xlabel('Time (ns)')
    ax_avgtrace1.set_ylabel('ADC Channels')
    # ax_dtraces.set_ylim(12500,13200)
    # plt.xlim(701,1299)
    plt.show()


  # Now ch 2
    wavecount2 = 0  # use this for averaging
    avgwave2 = np.zeros(len(traces[0]))  # grab trace array shape
    for i in range(len(channel)):
        # only interested in channel 0 rn
        if channel[i] != 2 or energy[i] < 50 or energy[i] == 4095: continue
        wavecount2 += 1
        # loop over trace and add to avgwave0
        for j in range(len(traces[i])):
            avgwave2[j] += traces[i][j]

    # now average the waveform
    for i in range(len(avgwave2)):
        avgwave2[i] = avgwave2[i] / wavecount2

    # Now subtract the baseline and flip, take the last 100 points
    avgwave2_bkg = np.average(avgwave2[-100:-1])
    for i in range(len(avgwave2)):
        avgwave2[i] = avgwave2[i] - avgwave2_bkg
        avgwave2[i] = -avgwave2[i]

    # Plot ch 2 avg trace
    # Plot avg trace traces 1 at a time
    plot_avgtrace2, ax_avgtrace2 = plt.subplots()
    ax_avgtrace2.plot(wavetime, avgwave2)
    ax_avgtrace2.set_xlabel('Time (ns)')
    ax_avgtrace2.set_ylabel('ADC Channels')
    # ax_dtraces.set_ylim(12500,13200)
    # plt.xlim(701,1299)
    plt.show()

    # Find the ratio of 2/0
    avgrat20 = np.zeros(len(avgwave0))
    for i in range(len(avgwave0)):
        avgrat20[i] = avgwave2[i]/avgwave0[i]
        if avgwave0[i] < 10: avgrat20[i] = 1

    # Modify channel 0 wave forms using this ratio (mult) and save
    modtrace_0 = []
    for i in range(len(traces)):
        if channel[i] != 0 or energy[i] < 50 or energy[i] == 4095: continue
        modwave = []
        tempwave = traces[i]
        tempbkg = np.average(tempwave[0:99])
        for j in range(len(tempwave)):
            tempwave[j] = -(tempwave[j] - tempbkg)
            modwave.append(tempwave[j] * avgrat20[j])

        modtrace_0.append(modwave)

    # Calculate the integral between 1000 and 5000 ns, 2 ns samp
    modint0 = []
    for i in range(len(modtrace_0)):
        modint0.append(np.sum(modtrace_0[i][500:2500]))
        print(modint0[i])

    # plot the modified waveform
    plot_modtrace0, ax_modtrace0 = plt.subplots()
    ax_modtrace0.plot(wavetime, modtrace_0[2])
    ax_modtrace0.set_xlabel('Time (ns)')
    ax_modtrace0.set_ylabel('ADC Channels')
    # ax_dtraces.set_ylim(12500,13200)
    # plt.xlim(701,1299)
    plt.show()


main()
