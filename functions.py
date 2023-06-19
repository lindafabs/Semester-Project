import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from scipy.optimize import brentq


#------------------------------------------------------------
# Time vector
#------------------------------------------------------------
def time_vector(fs, dur):
    '''
    :param fs: sampling frequency
    :param dur: duration of the signal
    :return: time vector
    '''
    T = 1/fs # sampling period
    t = np.linspace(0,dur, int(np.ceil(dur/T)))
    return t

#------------------------------------------------------------
# Class for different types of functions
#------------------------------------------------------------
class signals_ex:
    def __init__(self, time_vector, signal_frequency):
        self.f = signal_frequency #Hz
        self.t = time_vector
    def sine(self, amp):
        return amp * np.sin(2 * np.pi * self.f * self.t)
    def cosine(self, amp):
        return amp * np.cos(2 * np.pi * self.f * self.t)
    def sawtooth_sig(self, amp, slope):
        return amp* signal.sawtooth(2 * np.pi * self.f * self.t, slope)

    def compound_wave(self, amp1, amp2, f1, f2):
        return amp1 * np.sin(2 * np.pi * self.f1 * self.t) + amp2*np.sin(2 * np.pi * self.f2 * self.t)


#------------------------------------------------------------
# Test functions
#------------------------------------------------------------
def test_func(x):
    f0 = 1.2 # Hz
    return 0.9 * np.cos(2 * np.pi * f0 * x)

#------------------------------------------------------------
# Mid-Tread Quantizer
#------------------------------------------------------------

class quantizer:
    # implements a uniform deadzone quantizer with M levels over the [-A, A] interval
    def __init__(self, M, A=1):
        assert M % 2 == 1, "only considering mid-thread quantizers"
        self.clip = A
        self.offset = (M - 1) // 2
        self.step = 2 * A / M 
        
    def qbin(self, x):
        # return the INDEX of the quantization bin, i.e. an integer in the range [0, M-1]

        if np.max(np.abs(x)) > self.clip:
            raise OverflowError
        return (np.floor(x / self.step + 0.5) + self.offset).astype(int)
    def qvalue(self, i):
        # quantization value for bin i
        return self.step * (i - self.offset)
    def quantize(self, x):
        # return the quantized value
        return self.qvalue(self.qbin(x))
    def qthreshold(self, ix, iy):
        # return the midpoint between quantization bins ix and iy
        assert abs(ix - iy) == 1, "trying to obtain the threshold across more than 2 quantization levels"
        ix = ix + (0.5 if iy > ix else -0.5)
        return (ix - self.offset) * self.step
    
#------------------------------------------------------------
# Amplitude Sampler
#------------------------------------------------------------
def amplitude_sampler(f, T, Q, gd=1000):

    # f : input function
    # T : function period
    # Q : quantizer
    # gd : points per period in initial sampling grid
    def shifted_f(x, offset):
        # wrapper to shift the input function 
        return f(x) - offset
    
    transitions, bins = [0], [Q.qbin(f(0))]   # transition times and associated quantization bin
    num_samples = T * gd                      # start with a regular sampling to estimate transition points
    
    for n in range(1, num_samples):
        t = n * T / num_samples
        i = Q.qbin(f(t))
        if i != bins[-1]:
            # a level transition has occurred. Find the exact time
            #  first determine the value for the amplitude crossing
            threshold = Q.qthreshold(bins[-1], i)
            #  then determine the exact crossing time between neighboring samples
            transitions.append(brentq(shifted_f, (n - 1) * T / num_samples, t, args=threshold))
            bins.append(i)
    return np.array(transitions), np.array(bins)

#------------------------------------------------------------
# Decompose pulses
#------------------------------------------------------------
def decompose(t, i, T):
    assert t[0] == 0, 'first transition should be at t=0'
    M = np.max(i)
    pulses = []

    for m in range(1, np.max(i) + 1):
        t_start = t[0]
        on = (i[0] >= m)  # --> i[0] is the first bin index (in the previous case it's the highest)

        for n in range(1, len(i)):
            if on and i[n] < m:
                on = False
                pulses.append((t_start, t[n]))
            elif not on and i[n] == m:
                on = True
                t_start = t[n]
        if on:
            pulses.append((t_start, T))
    return pulses

#------------------------------------------------------------
# Plot decomposition
#------------------------------------------------------------
def plot_decomposition(pulses, Q, plot, points=1000):
    T = max(list(sum(pulses, ())))
    x = np.zeros(points)
    for p in pulses:
        n = np.round((points / T) * np.array(p)).astype(int)
        x[n[0]:n[1]] += 1
    if plot:
        plt.plot(np.linspace(0, T, points), Q.qvalue(x), label='Reconstructed quantized')
    return Q.qvalue(x)


#----------------------------------------------------
# Fourier Series
#----------------------------------------------------
def FS(n, t0,t1, T, t, delta):
    '''
    :param n: number of harmonics
    :param t0: start time instant
    :param t1: end time instant
    :param T: signal period
    :param x: time vector
    :param delta: step size

    :return: one-sided Fourier sum for one step function
    '''
    F=0
    for i in range(1,n):
        c = 1/(np.pi*i) * (np.exp(-1j*2*np.pi*i/T*(t0+t1)/2)*np.sin(2*np.pi*i/T*(t1-t0)/2))
        F_tmp  = delta * c*np.exp(1j*2*np.pi*i*t/T)
        F = F + F_tmp
    return F


#------------------------------------------------------------
# Run amplitude sampler
#------------------------------------------------------------
def amp_smp(func, T, q, xlim, k, plot):


    t_inst, q_idx = amplitude_sampler(func, T, q)
    pulse_times = decompose(t_inst, q_idx, T)
    x = np.linspace(0, T, 1000)  # time vector

    FS_complete = 0
    for i in range(len(pulse_times)):
        F_tmp = FS(k, pulse_times[i][0], pulse_times[i][1], T, x, q.step * 2)
        FS_complete = FS_complete + F_tmp
    #----------------------------------------------------------------
    off = np.real(FS_complete) - q.quantize(func(x))
    if plot:
        plt.plot(x, FS_complete - off, label="Fouries series")
        plt.plot(x, func(x), label="Original signal")
        plot_decomposition(decompose(t_inst, q_idx, T), q, plot= True)
        plt.title("Fourier series reconstruction")
        plt.xlim(0, xlim)
        plt.legend()
        plt.grid()

    return FS_complete, FS_complete- off
#------------------------------------------------------------
# Binary encoder
#------------------------------------------------------------
class binary_encoder:
    # Class to encode the index of the quantization bin into a binary string
    def __init__(self, NQbits):
       self.qbits = NQbits
    def encodeAll(self, bin_index):
        # Convert to binary string, removing the '0b' prefix
        # Fill with zeros to have a fixed length of Nqbits
        binary_list = []

        for i in range(0,len(bin_index)):
            binary_str = bin(bin_index[i])[2:].zfill(self.qbits) 
            binary_list.append(binary_str)

        return binary_list
    def bit_extract(bi_list, bit_pos ):
        return  [bits[bit_pos] for bits in bi_list]





