import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def plot_wave(x, y, xlim, ylim, label, col):

    # x lim = duration / n_periods]
    plt.plot(x, y, color=col, linewidth=1, label = label)
    plt.xlim([0, xlim])
    plt.ylim([-ylim, ylim])
    plt.xlabel('time [s]', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.grid()

def fourier_analysis(x, fsmp_ct):
    # Normalized Fourier transform of simple sine wave
    X=np.fft.fft(x)
    X/=np.abs(X).max()
    # Frequency vector
    N = len(X)
    n = np.arange(N)
    T = N / fsmp_ct
    freq = n / T
    #freq = np.fft.fftfreq(len(x_ct), 1 / fs_ct)

    return freq, X

def fourier_plot(freq, X, freq_lim, title):
    plt.plot(freq, np.abs(X), 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, freq_lim)
    plt.ylim(0, 1)
    plt.title(title)


def fourier_plot_db(freq, X, freq_lim, ylim, title):
    plt.plot(freq, 20*np.log10(np.abs(X)), 'b')
    plt.xlabel('Freq [Hz]')
    plt.ylabel('FFT Amplitude [dB]')
    plt.xlim(0, freq_lim)
    plt.ylim(ylim, 10)
    plt.title(title)