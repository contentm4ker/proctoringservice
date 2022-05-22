import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import *
from kalman import KalmanFilter

VIDEO_TO_ANALIZE_FILE_NAME = 'data/some_speech.mp4'
TMP_AUDIO_FILE_NAME = 'audio.wav'

def convert_to_decibel(val):
    ref = 1
    if val == 0.0:
        return 20 * np.log10(0.0001)
    else:
        return 20 * np.log10(abs(val) / ref)

video = VideoFileClip(VIDEO_TO_ANALIZE_FILE_NAME)
audio = video.audio
audio.write_audiofile(TMP_AUDIO_FILE_NAME)

if __name__ == '__main__':
    os.remove(TMP_AUDIO_FILE_NAME)
    k = KalmanFilter(0.05, 5)
    duration = video.duration
    time = []
    volume = []
    f_volume = []
    step = 0.025
    for t in range(int(duration / step)): # runs through audio/video frames obtaining them by timestamp with step 250 msec
        t = t * step
        if t > audio.duration or t > video.duration: break
        audio_frame = audio.get_frame(t) # numpy array representing mono/stereo values
        mono = audio_frame[0]
        val = 0.0 if np.abs(mono) < 1e-4 else mono
        val_db = convert_to_decibel(val)
        
        volume.append(val_db)
        f_volume.append(k.filter(val_db))
        time.append(t)

    plt.tight_layout()
    plt.axhline(y=0, color='r', linestyle='-')
    plt.plot(time, volume, label='Исходный сигнал')
    plt.plot(time, f_volume, label='Отфильтрованный сигнал')
    plt.xlabel('Время, с')
    plt.ylabel('Громкость, дБ')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
