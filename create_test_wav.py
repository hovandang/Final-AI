from pydub import AudioSegment
import math
import splitwav

file = 'dan_bau2.wav'
folder = './'
split_wav = splitwav.SplitWavAudioMubin(folder, file)
split_wav.multiple_split(min_per_split=15)