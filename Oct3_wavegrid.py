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
        data = np.genfromtxt(name, skip_header = True, delimiter =';', usecols = (0, 1, 2, 3, 4, 5, 6),converters={5: lambda s: float.fromhex(s)})
    else:
        data = np.genfromtxt(name, skip_header = True, delimiter =';', max_rows = 10000, usecols = (0, 1, 2, 3, 4, 5, 6),converters={5: lambda s: float.fromhex(s)})

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
def GetFile(runnum):
    # file path, use 'Path' function to make compatible with windows/mac
    # Data files

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
    file = r'pbeam_oct3_run' + str(runnum) + r'/RAW/SDataR_pbeam_oct3_run' + str(runnum) + r'.csv'

    return filepath / file, file

# Event class for time matched stuff
class FiberEvent:
    # constructor
    def __init__(self):
        self.energies0 = []
        self.energies2 = []
        self.energies4 = []
        self.times0 = []
        self.times2 = []
        self.times4 = []
        self.channels0 = []
        self.channels2 = []
        self.channels4 = []
        self.traces0 = []
        self.traces2 = []
        self.traces4 = []

    # add event
    def addevt(self,elist,tlist,chlist,tracelist,i0,i2,i4):
        if i0 != -1:
            self.energies0.append(elist[i0])
            self.times0.append(tlist[i0])
            self.channels0.append(chlist[i0])
            self.traces0.append(tracelist[i0])
        else:
            self.energies0.append(0)
            self.times0.append(0)
            self.channels0.append(0)
            self.traces0.append(0)

        if i2 != -1:
            self.energies2.append(elist[i2])
            self.times2.append(tlist[i2])
            self.channels2.append(chlist[i2])
            self.traces2.append(tracelist[i2])
        else:
            self.energies2.append(0)
            self.times2.append(0)
            self.channels2.append(0)
            self.traces2.append(0)
        if i4 != -1:
            self.energies4.append(elist[i4])
            self.times4.append(tlist[i4])
            self.channels4.append(chlist[i4])
            self.traces4.append(tracelist[i4])
        else:
            self.energies4.append(0)
            self.times4.append(0)
            self.channels4.append(0)
            self.traces4.append(0)

    def pickchan(self,chan):
        if chan == 0:
            return self.energies0,self.times0,self.channels0,self.traces0
        elif chan == 2:
            return self.energies2,self.times2,self.channels2,self.traces2
        elif chan == 4:
            return self.energies4,self.times4,self.channels4,self.traces4
        else:
            print('Picked bad channel for builtevent pickchan(), abort')
            sys.exit()

    # can play this trick to get the number of built events
    # only works since built event length is universal
    def __len__(self):
        return len(self.energies0)

def main(runnum):
    # Read the CAEN data file in csv format
    # Different files available to process
    ###################################################

    # Remember to change your file path for your computer

    ###################################################
    # Constant filepath variable to get around the problem of backslashes in windows
    # The Path library will use forward slashes but convert them to correctly treat your OS
    # Also makes it easier to switch to a different computer
    datafile, filecsv = GetFile(runnum)
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
    wavesdat = True
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
        else: traces.append([0,0])

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

    # Time match channels into events
    # Can be complete (3) or incomplete (1 or 2)
    # evts driven by channels 0 and 2, no event that's only OF (4)
    builtevent = FiberEvent()

    ch02_e = [] # pairs relative to 0
    ch02_t = []
    ch12_e = [] # pairs relative to 1
    ch12_t = []

    # Loop over and look for neighboring channels with close times
    # 1.32 ms dump spacing, so pretty wide window
    twindow = 50000000 # 50 us window I believe

    # for storing used indices, this is because I must search the entire
    # span of channels each time
    used_i = []

    i = 0
    while i < len(channel) - 1:
        # Make sure to get out if at end
        if i == len(channel) - 1: break
        if i in used_i:
            i+=1
            continue
        if energy[i] < 1 or energy[i] == 4095:
            i += 1
            continue
        match = False

        indices = [-1,-1,-1]

        # put channel[i] in proper spot ( /2 to get 0,1, or 2)
        indices[int(channel[i]/2)] = i

        # for modifying i
        incr = 0

        # Must look through whole array due to memory problems
        # Sometimes stuff will get dumped in bulk before it should
        for j in range(i+1,len(channel)):
            # break if at end
            if j == len(channel): break

            # some threshold needed
            if energy[j] < 1 or energy[j] == 4095: continue

            if channel[i] == channel[j]: continue # if same channel num
            if abs(time[i] - time[j]) > twindow: continue # if outside time window, time ordered data
            if j in used_i: continue

            match = True

            # get index by div / 2
            indices[int(channel[j] / 2)] = j
            used_i.append(j)
            incr += 1


        # if matched, do not increment again
        if match == True:
            builtevent.addevt(energy, time, channel, traces, indices[0], indices[1], indices[2])
            i += 1
            continue
        elif match != True and channel[i] != 4:
            builtevent.addevt(energy, time, channel, traces, indices[0], indices[1], indices[2])
            i += 1
            continue
        else: # don't add if only channel 4 (optical)
            print(time[i], ' ', channel[i], ' ', energy[i])
            i += 1
            continue


    ############################################################
    # Plotting traces

    # make a window plot showing each real traces
    # Just plot 20 traces

    xStart = 0
    xEnd = len(traces[1]) - 1
    wavetime = np.linspace(xStart, xEnd, len(traces[1]))
    # Convert sample number to time
    # CAEN samples every 2 ns
    wavetime = wavetime * 2

    numcols = 4
    numrows = 5

    windowsize = (20,20)

    fig_multtrac, ax_multtrac = plt.subplots(numrows,numcols,sharex=True,sharey=True, figsize=windowsize,layout='compressed')
    axes_flat = ax_multtrac.flatten()

    # choose a channel to plot (0, 2, or 4)
    pickch = 4
    wgE,wgT,wgCh,wgTr = builtevent.pickchan(pickch)
    goodtr = []

    wgindex = 0

    for i, ax in enumerate(axes_flat):
        if i < len(builtevent):

            if wgE[wgindex] == 0:
                wgindex += 1
                continue # skip if empty

            goodtr.append(wgTr[wgindex])

            pulselabel = 'Pulse ' + str(wgindex)
            ax.set_box_aspect(1)
            ax.plot(wavetime, wgTr[wgindex])
            ax.text(0.95,0.05,pulselabel,transform=ax.transAxes,ha='right',va='bottom',)
            #ax.legend(loc='lower right')
            wgindex += 1

            axes_flat[i].label_outer()
        else:
            axes_flat[i].remove()


    # Manually re-enable labels for plots that have nothing below them

    for i in range(len(goodtr)):
        # If the index plus one column width is outside the plot count, it's a bottom plot
        if i + 4 >= len(goodtr):
            axes_flat[i].tick_params(labelbottom=True)
    fig_multtrac.canvas.draw()
    xaxis_shift = 0
    last_ax = axes_flat[len(goodtr) -1]
    renderer = fig_multtrac.canvas.get_renderer()
    bbox = last_ax.get_tightbbox(renderer)
    xaxis_shift = fig_multtrac.transFigure.inverted().transform((0, bbox.y0))[1] - 0.02

    fig_multtrac.supxlabel('Time (ns)',fontsize=30,y=xaxis_shift)
    fig_multtrac.supylabel('ADC Channels',fontsize=30)
    plt.xlim(0,10000)
    plt.ylim(14000,16000)
    #plt.subplots_adjust(wspace=0.1, hspace=0.1)

    figgrid_name = savename + '_wavegrid_Oct3_ch' + str(pickch)
    plt.savefig(figurepath / (figgrid_name + '.eps'), format='eps')
    plt.savefig(figurepath / (figgrid_name + '.png'), format='png')
    plt.savefig(figurepath / (figgrid_name + '.svg'), format='svg')
    plt.show()



runnum = 23
main(runnum)