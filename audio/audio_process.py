import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import *
from kalman import KalmanFilter

VIDEO_TO_ANALIZE_FILE_NAME = 'data/keyboard_disable_mic_voice.mp4'
TMP_AUDIO_FILE_NAME = 'audio.wav'

MIN_VOLUME = 20 * np.log10(0.0001)
TOO_LOUD_VOLUME = -50

def convert_to_decibel(val):
    ref = 1
    if val == 0.0:
        return MIN_VOLUME
    else:
        return 20 * np.log10(abs(val) / ref)

def find_segments(arr, cond, l = 10):
    result = []
    ind_1 = -1
    for i in range(len(arr)):
        if cond(arr[i]):
            if ind_1 < 0:
                ind_1 = i
        else:
            if ind_1 > 0 and i - ind_1 > l:
                result.append([ind_1, i])
            ind_1 = -1

    # for last segment
    if ind_1 > 0 and i - ind_1 > l:
        result.append([ind_1, i])

    return result

def merge_segments(segments):
    result = []
    start, end = segments[0]
    for s in segments[1:]:
        if s[0] - end <= 1:
            end = s[1]
        else:
            result.append([start, end])
            start, end = s
    result.append([start, end])
    return result

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

    # find segments w/ no sound
    min_vol_cond = lambda x: x == MIN_VOLUME
    no_sound_segments = find_segments(volume, min_vol_cond)
    no_sound_segments_t = [[time[s[0]], time[s[1]]] for s in no_sound_segments]
    if len(no_sound_segments_t) > 0:
        print('No sound detected', no_sound_segments_t)

    # find segments w/ too loud volume
    too_loud_cond = lambda x: x > TOO_LOUD_VOLUME
    too_loud_segments = find_segments(f_volume, too_loud_cond, 20)
    too_loud_segments_t = merge_segments([[round(time[s[0]], 4), round(time[s[1]], 4)] for s in too_loud_segments])
    if len(too_loud_segments_t) > 0:
        print('Too loud volume detected', too_loud_segments_t)

    plt.tight_layout()
    plt.axhline(y=0, color='r', linestyle='-')
    plt.plot(time, volume, label='Исходный сигнал')
    plt.plot(time, f_volume, label='Отфильтрованный сигнал')
    plt.xlabel('Время, с')
    plt.ylabel('Громкость, дБ')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
