import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from scipy.optimize import brentq
import functions, utils


# ------------------------------------------------------------
# Amplitude Sampler
# ------------------------------------------------------------
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
# Fourier series sum
#------------------------------------------------------------
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
# pipe-line of amplitude sampler
#------------------------------------------------------------

def ampSmp_run(t_range,T,q, test_function):

    t_inst, q_idx = amplitude_sampler(test_function, T, q)
    pulse_times = functions.decompose(t_inst, q_idx, T)

    # Fourier sum
    FS_complete = 0
    for i in range(len(pulse_times)):
        F_tmp = FS(200, pulse_times[i][0], pulse_times[i][1], T, t_range, q.step * 2)
        FS_complete = FS_complete + F_tmp

    #plot

    offset = FS_complete - functions.plot_decomposition(pulse_times, q)

    plt.plot(t_range, FS_complete - offset, 'b', label="Fouries series")
    plt.plot(t_range, test_function(t_range), 'black', label="Original signal")
    functions.plot_decomposition(pulse_times, q)
    plt.title("Fourier series reconstruction")
    plt.legend()
    plt.grid()
    plt.show()

    # Fourier frequency analysis
    freq_FS, X_FS = utils.fourier_analysis(FS_complete, fsmp=100)
    plt.figure(figsize=(7, 6))
    utils.fourier_plot(freq_FS, X_FS, freq_lim=10, title="Frequency spectrum of Fourier sum")
    plt.show()