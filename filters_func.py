import numpy as np 
import scipy as sp 
from scipy.signal import butter,filtfilt
import matplotlib.pyplot as plt
import utils

class FIR_class: 
    # Implements multiple types of filters with filer taps K and time delay Td
    def __init__(self, taps, delay):
        self.K = taps
        self.Td = delay

    def movingAvg(self, x, q_sig):
        '''
        Moving average filter
        :param x: time vector
        :param q_sig: original quantized signal
        :return: filtered signal
        '''
        y=np.zeros(len(x))
        for k in range(1, self.K):
            a_k = (1/self.K)
            q_delay=np.concatenate((np.zeros( self.Td*k), q_sig[:-k*self.Td]))
            y = y + a_k*q_delay

        corr = q_sig.max() / y.max() # amplitude correction
        print(corr)
        return y*corr

    def hamming(self, x, q_sig):
        '''
        Hamming window  filter
        :param x: time vector
        :param q_sig: original quantized signal
        :return: filtered signal
         '''
        y=np.zeros(len(x))
        for k in range(1,self.K):
            a_k = ( 0.54 - 0.46 * np.cos((2*np.pi*k)/(self.K-1)))
            q_delay=np.concatenate((np.zeros(self.Td*k), q_sig[:-k*self.Td]))
   
            y = y + a_k*q_delay

        corr = q_sig.max()/y.max() # amplitude correction
        print(corr)
        return y*corr
    
    def bartlett(self, x, q_sig):
        '''
         Bartlett triangular filter
         :param x: time vector
         :param q_sig: original quantized signal
         :return: filtered signal
         '''
        y=np.zeros(len(x)-1)
        for k in range(1,self.K):
            a_k = ( (2/self.K) * ((self.K-1)/2 - abs((self.K-1)/2 - k)) )
            q_delay=np.concatenate((np.zeros(self.Td*k-1), q_sig[:-k*self.Td]))
            y = y + a_k*q_delay

        corr = q_sig.max() / y.max() # amplitude correction
        print(corr)
        return y*corr
    

#------------------------------------------------------------
# Butterworth filter
#------------------------------------------------------------


def butter_lowpass_filter(data, cutoff, fs, order, nyq):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    print(len(b)), print(len(a))
    y = filtfilt(b, a, data)
    return y


#------------------------------------------------------------
# Lowpass
#------------------------------------------------------------

def low_pass_fir(signal, cutoff_freq, num_taps, f_smp):
    # Normalize cutoff frequency
    normalized_cutoff = cutoff_freq / (0.5 * f_smp)

    # Compute filter coefficients using the windowing method
    coefficients = np.sinc(2 * normalized_cutoff * (np.arange(num_taps) - (num_taps - 1) / 2))*0.025
    window = np.hamming(num_taps)
    coefficients = coefficients * window

    # Apply the filter to the signal using convolution
    filtered_signal = np.convolve(signal, coefficients, mode='same')

    return filtered_signal

#------------------------------------------------------------
# Filter performance
#------------------------------------------------------------

def perform(test_sig, og_sig, filter_name):
    '''
    Calculates the performance of the filter with Means square error and SNR parameters
    :param test_sig: filtered signal
    :param og_sig: original signal
    :param filter_name: type of filter used
    :return: Mean square error value, SNR value
    '''
    mse = np.abs(np.mean((og_sig - test_sig) ** 2))
    snr = np.abs(10 * np.log10(np.sum(og_sig ** 2) / np.sum((og_sig- test_sig) ** 2)))

    print('MSE - {} filter = '.format(filter_name), mse)
    print('SNR - {} filter = '.format(filter_name), snr)


def filters_plot(filter_name, x, y,x_smp,y_smp, y_og, y_og_smp):
    # ----------------------------------------------------------
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, y, label='{}'.format(filter_name))
    plt.plot(x, y_og, label='Original signal')
    plt.title('{}'.format(filter_name), fontsize=9)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x_smp, y_smp, label='{} - smp'.format(filter_name))
    plt.plot(x_smp, y_og_smp, label='Original signal')
    plt.title('{} - sampled'.format(filter_name), fontsize=9)
    plt.legend()
    # ------------------------------------------------------------
    perform(y, y_og, '{}'.format(filter_name))
    perform(y_smp, y_og_smp, '{} - smp'.format(filter_name))

def filters_plot_fourier(freq_FS, X_FS, freq_FS_smp, X_FS_smp, freq_limit, fir_name):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    utils.fourier_plot(freq_FS, X_FS, freq_lim=freq_limit, title="Frequency spectrum of {}".format(fir_name))
    plt.subplot(1, 2, 2)
    utils.fourier_plot(freq_FS_smp, X_FS_smp, freq_lim=freq_limit, title=f"Frequency spectrum of {fir_name}- smp".format(fir_name))
    plt.tight_layout()