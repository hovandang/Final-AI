from os import path
from pydub import AudioSegment

# files                                                                         
src = "sao_truc6.mp3"
dst = "sao_truc6.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")