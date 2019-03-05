import numpy as np
from scipy import signal, stats
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
#import h5py
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

def get_filter_coefs(fs):
    
    # assemble the filter b,a coefficients:
    coefs = []

    '''# bandpass filter parameters
	Original parameters
    lowcut=43
    highcut=260'''
    lowcut=50
    highcut=350
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

# get filter coefficients
coefs = get_filter_coefs(fs)
# generate random gaussian "data"
data = np.random.randn(128*fs)

# filter it:
resp = filter_data(data,coefs)

# compute the amplitude spectral density (ASD) of the original data, and the filtered data:
NFFT = fs/2
Pxx_data, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT)
Pxx_resp, freqs = mlab.psd(resp, Fs = fs, NFFT = NFFT)

# The asd is the square root; and let's normalize it to 1:
norm = np.sqrt(Pxx_data).mean()
asd_data = np.sqrt(Pxx_data)/norm
asd_resp = np.sqrt(Pxx_resp)/norm

# get the predicted filter frequency response using signal.freqz:
Nc = 2000
filt_resp = np.ones(Nc)
for coef in coefs:
    b,a = coef
    w,r = signal.freqz(b,a,worN=Nc)
    filt_resp = filt_resp*np.abs(r)
freqf = (fs * 0.5 / np.pi) * w
# We "double pass" the filtering using filtfilt, so we square the filter response
filt_resp = filt_resp**2

strain_H1_filt = filter_data(strain_H1, coefs)
strain_L1_filt = filter_data(strain_L1, coefs)

tau_max=0.01    #10 ms
tau_max_index=int(np.around(tau_max/dt))

#Caluclating the mean batchwise cross correlations, instants with maximum cross correlation in a batch and the tau at which it occurs
def CD_n_cmax_batchwise_full(window_duration):
	window_index_len=int(window_duration/dt)
	if window_index_len%2==1:    #Helps for computing the middlemost instant in the batch with ease
		window_index_len+=1

	index_before_start=int(30/dt)+1
	index_after_end=np.size(time)-1-int(30/dt)
	batches_index=np.arange(tau_max_index+index_before_start, index_after_end-tau_max_index-window_index_len, window_index_len)    #Indices of
	batch_time=time[batches_index+window_index_len/2]    #Mid instances in every batch time before the event

	#These are uniform with a spacing of the window length
	no_batches=np.size(batches_index)    #No. of batches
	C_tau=np.empty(2*tau_max_index+1)	#Cross correlations at every tau
	C_max=np.empty(no_batches)    #Maximum cross correlations in every batch
	C_max_tau=np.empty(no_batches)    #tau of occurrence of maximum cross correlation in every batch

	#Calculating D just by using a average
	k=0
	for i in batches_index:
		for j in np.arange(-tau_max_index, tau_max_index+1):
		    C_tau[j+tau_max_index]=stats.pearsonr(strain_H1_filt[i+j:i+window_index_len+j], strain_L1_filt[i:i+window_index_len])[0]
		C_max_index=np.argmax(np.abs(C_tau))
		C_max[k]=C_tau[C_max_index]
		C_max_tau[k]=(C_max_index-tau_max_index)*dt
		k+=1

	C_max_overall_index=np.argmax(np.abs(C_max))
	print 'The maximum absolute cross correlation using bandpassing and notching is', C_max[C_max_overall_index], 'and occurs in the batch with mid-instant', batch_time[C_max_overall_index], 'with a corresponding tau of', C_max_tau[C_max_overall_index]


	fac=int(20*0.1/window_duration)
	C_max_short_size=np.size(C_max)/fac
	C_max_short=np.empty(C_max_short_size)
	for i in np.arange(C_max_short_size):
		C_max_index=np.argmax(np.abs(C_max[fac*i:fac*(i+1)]))
		C_max_short[i]=C_max[fac*i+C_max_index]
	batch_time_centred=batch_time[fac/2::fac]-tevent
	plt.figure()
	plt.plot(batch_time_centred, C_max_short, '.')
	plt.xlabel(r'$t-t_{event}\ (s)$')
	plt.ylabel(r'$C_{max}(t)$')
	plt.title(r'$Max.\ CC\ for\ different\ mid\ instants\ of\ the\ '+str(window_duration)+'\ s\ batches$')
	plt.axvline(-0.02, color='r', linestyle='--')
	plt.axvline(0.02, color='r', linestyle='--')
	plt.savefig('C_max_vs_batch_time_w_point'+str(int(10*window_duration))+'_full.png')

CD_n_cmax_batchwise_full(0.1)
CD_n_cmax_batchwise_full(0.2)

