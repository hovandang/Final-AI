import librosa
import librosa.display as dsp
from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np
src = './sao_truc6.wav'
sample,sample_rate = librosa.load(src,sr = None)
print('Total number of samples: ',sample.shape[0])
print('Sample rate: ',sample_rate)
print('Lenngth of file in seconds: ',librosa.get_duration(sample))

# fig, ax = plt.subplots(nrows=2, sharex=True,figsize=(10,7))
# librosa.display.waveshow(sample, sr=sample_rate, ax=ax[0])
# ax[0].set(title='Envelope view, mono')
# ax[0].label_outer()
# librosa.display.waveshow(sample1, sr=sample_rate1, ax=ax[1])
# ax[1].set(title='Envelope view, stereo')
# ax[1].label_outer()
i=0
while (i<(len(sample)/(sample_rate*5))):
    d = librosa.stft(sample[sample_rate*5*i:sample_rate*5*(i+1)])
    D = librosa.amplitude_to_db(np.abs(d),ref=np.max)
    dsp.specshow(D,y_axis='log',x_axis='s',sr=sample_rate,)
    # plt.ylim(0, 180000)
    plt.axis('off')
    plt.savefig('./10/sao_truc'+str(i+152)+'.png', dpi=200, bbox_inches='tight',transparent=True, pad_inches=0)
    i = i+1