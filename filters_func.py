import numpy as np 
import scipy as sp 
from scipy import signal
from scipy.signal import butter,filtfilt

class FIR_class: 

    def __init__(self, taps, delay):
        self.K = taps
        self.Td = delay

    def movingAvg(self, x, q_sig, corr):
        y=np.zeros(len(x)-1)
    
        for k in range(1, self.K):
            a_k = (1/k)*corr
            q_delay=np.concatenate((np.zeros( self.Td-1), q_sig[:-k*self.Td]))
            y = y + a_k*q_delay  

        return y
    
    def movingAvg_weigth(self, x, q_sig, corr):
        y=np.zeros(len(x)-1)

        for k in range(1,self.K):
            a_k = (1/(k*(k+1)))*corr
            q_delay=np.concatenate((np.zeros(self.Td*k-1), q_sig[:-k*self.Td]))
   
            y = y + a_k*q_delay  
        return y 
    
    def hamming(self, x, q_sig, corr):
        y=np.zeros(len(x)-1)

        for k in range(1,self.K):
            a_k = ( 0.54 - 0.46 * np.cos((2*np.pi*k)/(self.K-1)))*corr
            q_delay=np.concatenate((np.zeros(self.Td*k-1), q_sig[:-k*self.Td]))
   
            y = y + a_k*q_delay  
        return y
    
    def bartlet(self, x, q_sig, corr):
        y=np.zeros(len(x)-1)

        for k in range(1,self.K):
            
            a_k = ( (2/self.K) * ((self.K-1)/2 - abs((self.K-1)/2 - k)) )*corr
            q_delay=np.concatenate((np.zeros(self.Td*k-1), q_sig[:-k*self.Td]))
   
            y = y + a_k*q_delay  
        return y
    

#------------------------------------------------------------
# Butterworth filter
#------------------------------------------------------------


def butter_lowpass_filter(data, cutoff, fs, order):
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

    mse = np.mean((og_sig - test_sig) ** 2)
    snr = 10 * np.log10(np.sum(og_sig ** 2) / np.sum((og_sig- test_sig) ** 2))

    print('MSE - {} filter = '.format(filter_name), mse)
    print('SNR - {} filter = '.format(filter_name), snr)