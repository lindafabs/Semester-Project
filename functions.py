import numpy as np
from scipy.optimize import brentq


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


class binEnc:
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
