import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from librosa import load,display
import scipy
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

# this function gets an audio file.wav and returns an audio plot
def display_WAV(wav_file):           
    # use librosa module for handling WAV audio files     
    f_amp,samp_rate = load(wav_file) 
    plt.figure
    display.waveplot(y = f_amp, sr = samp_rate)
    plt.xlabel('Time [Sec.]')
    plt.ylabel('Amplitude')
    return plt.show()

# convert an audio signal into a fast fourier transform representation
def WAV2FFT(wav_file):
    f_amp,samp_rate = load(wav_file)  # load audio file
    n = len(f_amp)                    # length of signal
    T = 1/samp_rate                   # trans. from continuous to descrete time
    mag = np.fft.fft(f_amp)           # convert signals magnitude using FFT
    # prepare FFT presentation
    freq = np.linspace(0.0,int(1.0/(2.0*T)),n//2) # generate a vector of frequencies
    fig,ax = plt.subplots()                       # setup a graph
    ax.plot(freq,2.0/n*np.abs(mag[:n//2]))
    plt.xlabel('Freq')
    plt.ylabel('Magnitude')
    return plt.show()

# convert an audio signal into a spectrogram image
def get_Spectrogram(wav_file,show_flag):
    f_amp,samp_rate = load(wav_file)          # load audio file 
    num_per_seg = int(samp_rate * 20 // 1000) # set a number of observations per sec. 
    # utilize a scipy module signal function to map signals spctrogram  
    fr,t,Sxx = scipy.signal.spectrogram(f_amp,          # audio amplitudes
                                        samp_rate,      # audio recording feature
                                        window=('hann'),# type of FFT conv. window
                                        nperseg=num_per_seg, # no. of samples
                                        noverlap=0.50*num_per_seg, # qty of overlaps
                                        mode='magnitude')
    max_fr=2000   # limit frquency upto 2000hz (higher has less contribution)
    if show_flag == 'Yes':
        # prepare plot and convert it to an image 
        plt.pcolor(t,fr,Sxx,shading='auto')
        plt.ylim(0,max_fr)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    
    idx = np.where(fr <= max_fr)[0][-1]+1
    im_data = np.log(Sxx[:idx,:]+1e-14)
    scaler = MinMaxScaler()
    scaler.fit(im_data)
    im_data = scaler.transform(im_data)
    img = Image.fromarray(cm.jet(im_data,bytes=True))
    img = img.convert('RGB')
    return img


