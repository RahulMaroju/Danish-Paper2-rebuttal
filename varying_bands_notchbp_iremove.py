# Calculate the cross correlation between the average cross correlation in (1280, 4050) s ignoring 60 s around the GW event and the GW event (C_gw) in by bandpass filtering in various sub-bands
import numpy as np
from scipy import signal, stats
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import readligo as rl

fn_H1 = 'H-H1_LOSC_4_V2-1126257414-4096.hdf5'
strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
fn_L1 = 'L-L1_LOSC_4_V2-1126257414-4096.hdf5'
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')

fs = 4096
time = time_H1
dt = time[1] - time[0]
tevent = 1126259462.422         # Mon Sep 14 09:50:45 GMT 2015 

# number of sample for the fast fourier transform:
NFFT = 1*fs
fmin = 10
fmax = 2000
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)

# We will use interpolations of the ASDs computed above for whitening:
psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)

# generate linear time-domain filter coefficients, common to both H1 and L1.
# First, define some functions:

# This function will generate digital filter coefficients for bandstops (notches).
# Understanding it requires some signal processing expertise, which we won't get into here.
def iir_bandstops(fstops, fs, order=4):
    """ellip notch filter
    fstops is a list of entries of the form [frequency (Hz), df, df2]                           
    where df is the pass width and df2 is the stop width (narrower                              
    than the pass width). Use caution if passing more than one freq at a time,                  
    because the filter response might behave in ways you don't expect.
    """
    nyq = 0.5 * fs

    # Zeros zd, poles pd, and gain kd for the digital filter
    zd = np.array([])
    pd = np.array([])
    kd = 1

    # Notches
    for fstopData in fstops:
        fstop = fstopData[0]
        df = fstopData[1]
        df2 = fstopData[2]
        low = (fstop - df) / nyq
        high = (fstop + df) / nyq
        low2 = (fstop - df2) / nyq
        high2 = (fstop + df2) / nyq
        z, p, k = iirdesign([low,high], [low2,high2], gpass=1, gstop=6,
                            ftype='ellip', output='zpk')
        zd = np.append(zd,z)
        pd = np.append(pd,p)

    # Set gain to one at 100 Hz...better not notch there                                        
    bPrelim,aPrelim = zpk2tf(zd, pd, 1)
    outFreq, outg0 = freqz(bPrelim, aPrelim, 100/nyq)

    # Return the numerator and denominator of the digital filter                                
    b,a = zpk2tf(zd,pd,k)
    return b, a

def get_filter_coefs(lowcut, highcut, fs):
    
    # assemble the filter b,a coefficients:
    coefs = []

    '''# bandpass filter parameters
	Original parameters
    lowcut=43
    highcut=260'''
    order = 4
    
    # bandpass filter coefficients 
    nyq = 0.5*fs
    low = lowcut / nyq
    high = highcut / nyq
    bb, ab = butter(order, [low, high], btype='band')
    coefs.append((bb,ab))

    # Frequencies of notches at known instrumental spectral line frequencies.
    # You can see these lines in the ASD above, so it is straightforward to make this list.
    notchesAbsolute = np.array(
        [14.0,34.70, 35.30, 35.90, 36.70, 37.30, 40.95, 60.00, 
         120.00, 179.99, 304.99, 331.49, 510.02, 1009.99])

    # notch filter coefficients:
    for notchf in notchesAbsolute:                      
        bn, an = iir_bandstops(np.array([[notchf,1,0.1]]), fs, order=4)
        coefs.append((bn,an))

    # Manually do a wider notch filter around 510 Hz etc.          
    bn, an = iir_bandstops(np.array([[510,200,20]]), fs, order=4)
    coefs.append((bn, an))

    # also notch out the forest of lines around 331.5 Hz
    bn, an = iir_bandstops(np.array([[331.5,10,1]]), fs, order=4)
    coefs.append((bn, an))
    
    return coefs

# and then define the filter function:
def filter_data(data_in,coefs):
    data = data_in.copy()
    for coef in coefs:
        b,a = coef
        # filtfilt applies a linear filter twice, once forward and once backwards.
        # The combined filter has linear phase.
        data = filtfilt(b, a, data)
    return data

# Calculating the cross correlation in a 0.1 s batch around the GW event
semi_window_duration=0.05
index_gw_window_start=int((tevent-time[0]-semi_window_duration)/dt)
index_gw_window_end=int((tevent-time[0]+semi_window_duration)/dt)
tau_max=0.01    #10 ms
tau_max_index=int(np.around(tau_max/dt))
tau_range=np.linspace(-tau_max_index*dt, tau_max_index*dt, 2*tau_max_index+1)
C_gw=np.empty(2*tau_max_index+1)

def findCgw_bandpass(lowcut, highcut):
	# get filter coefficients
	coefs = get_filter_coefs(lowcut, highcut, fs)
	strain_H1_filt = filter_data(strain_H1, coefs)
	strain_L1_filt = filter_data(strain_L1, coefs)
	for i in np.arange(-tau_max_index, tau_max_index+1):
		C_gw[i+tau_max_index]=stats.pearsonr(strain_H1_filt[index_gw_window_start+i:index_gw_window_end+i], strain_L1_filt[index_gw_window_start:index_gw_window_end])[0]
	return strain_H1_filt, strain_L1_filt, C_gw

def CDcorrelation(lowcut, highcut, t, window_duration):
	#Calculating the cross correlation between CC and D
	#t: contains intervals
	strain_H1_filt, strain_L1_filt, C_gw=findCgw_bandpass(lowcut, highcut)
	window_index_len=int(window_duration/dt)
	D_t1_t2=np.zeros(2*tau_max_index+1)
	batches_index_t1_t2=np.empty(0, int)
	for i in np.arange(np.size(t)/2):
		t1_index=int(t[2*i]/dt)+1
		t2_index=int(t[2*i+1]/dt)
		batches_index_t1_t2=np.append(batches_index_t1_t2, np.arange(tau_max_index+t1_index, t2_index-tau_max_index-window_index_len, window_index_len))
	C_tau=np.empty(2*tau_max_index+1)	#Cross correlations at every tau
	for i in batches_index_t1_t2:
		for j in np.arange(-tau_max_index, tau_max_index+1):
		    C_tau[j+tau_max_index]=stats.pearsonr(strain_H1_filt[i+j:i+window_index_len+j], strain_L1_filt[i:i+window_index_len])[0]
		D_t1_t2+=C_tau
	D_t1_t2/=np.size(batches_index_t1_t2)
	E=stats.pearsonr(C_gw, D_t1_t2)[0]
	print str(lowcut)+'-'+str(highcut)+') Hz:', E

print 'E(-10 ms, 10 ms) in various sub-bands from (1280, 1988)U(2108, 4050) s:'
CDcorrelation(30, 50, [1280, 1988, 2108, 4050], 0.1)
CDcorrelation(50, 70, [1280, 1988, 2108, 4050], 0.1)
CDcorrelation(70, 90, [1280, 1988, 2108, 4050], 0.1)
CDcorrelation(90, 110, [1280, 1988, 2108, 4050], 0.1)
CDcorrelation(110, 130, [1280, 1988, 2108, 4050], 0.1)
CDcorrelation(130, 150, [1280, 1988, 2108, 4050], 0.1)

