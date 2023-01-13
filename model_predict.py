from keras.models import load_model
import keras.utils as image
import matplotlib.pyplot as plt
import numpy as np
import os
import splitwav
import librosa
import librosa.display as dsp
from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":

    # load model

    model = load_model("model_music_instrument.h5")
    folder = './test'
    file = 'dan_trung3.wav'
    sample,sample_rate = librosa.load(folder+'/'+file,sr = None)
    k = librosa.get_duration(sample)
    split_wav = splitwav.SplitWavAudioMubin(folder, file)
    split_wav.multiple_split(min_per_split=10)
    i = 0 
    while (i+10<60):
        src = folder + '/'+ str(i)+'_' + file
        sample,sample_rate = librosa.load(src,sr = None)
        d = librosa.stft(sample)
        D = librosa.amplitude_to_db(np.abs(d),ref=np.max)
        dsp.specshow(D,y_axis='log',x_axis='s',sr=sample_rate)
        plt.axis('off')
        plt.savefig(src.rstrip('.wav')+'.png', dpi=200, bbox_inches='tight',transparent=True, pad_inches=0)
    # image path
        img_path = src.rstrip('.wav')+'.png'   
        print (img_path)
    # load a single image
        new_image = load_image(img_path)

    # check prediction
        pred = model.predict(new_image)
        print (pred)
        i = i+10