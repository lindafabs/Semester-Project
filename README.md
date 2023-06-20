## Semester-Project - EPFL
This repository contains the research work done during the spring semester of 2023 for my semester project in Electrical Engineering at EPFL. 
The project, done in collaboration with the research department of the company Bang & Olufsen, consists in the investigation of the **Continuous-time paradigm in Digital Signal Processing and Neural Networks**, inspired from the work of Prof. Yannis Tsividis. 

Continuous-Time Digital Signal Processing is a signal processing paradigm that allows the discretization of a signal, working only by means of amplitude quantization*. The discrete signal remains continuous in time in the digital domain, allowing a number of advantages from a digital processing point of view, such as removed aliasing phenomena
and reduction of quantization error. It has been shown how this method can be achieved with “event-driven” control systems using level-crossing sampling, amplitude quantization and input signal decomposition. [1] [2] [3] In particular, since it was shown that the continuous-time discretization can be described in terms of FIR- IIR-like filters, an exploration of this method can be done in the domain of Convolutional Neural Networks, exploiting the FIR filter nature of said nets.

The repository is organized as follows: 
- [Here](https://github.com/lindafabs/Semester_Project/blob/main/Final_notebook.ipynb) you can find the Jupyter notebook for the complete and descriptive summary of the project 
- [Quantization_contd](https://github.com/lindafabs/Semester_Project/blob/main/quantization_contd.ipynb) contains additional tests for section 1
- [Amplitude_sampling_contd](https://github.com/lindafabs/Semester_Project/blob/main/amplituteSampling_contd.ipynb) contains additional tests for section 3
- [FIR_contd](https://github.com/lindafabs/Semester_Project/blob/main/FIR_contd.ipynb) contains additional tests for sections 4


### Student and Advisors
Student: Linda Fabiani\
Supervisor: Paolo Prandoni - LCAV (paolo.prandoni@epfl.ch)\
Scientific Assistant Contacts: Pablo Martinez-Nuevo (B&O) PMN@bang-olufsen.dk, Martin Bo Møller (B&O) MIM@Bang-Olufsen.dk

### References
[1] Y. Tsividis, " Continuous-time digital signal processing.," Electronics Letters, pp. 39(21), p.1., 2003\
[2] Y. Tsividis, "Event-driven data acquisition and continuous-time digital signal processing," IEEE Custom Integrated Circuits Conference 2010, pp. (pp. 1-8),
2010, September.\
[3] Y. Tsividis, "Event-driven data acquisition and digital signal processing—A tutorial.," IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 57 (8),
pp. 577-581., 2010.\
