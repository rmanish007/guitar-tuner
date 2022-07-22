import numpy as np
import pyaudio

# import time

NOTE_MIN = 40  # E2
NOTE_MAX = 64  # E4
FSAMP = 22050  # Sampling frq in hz
FRAME_SIZE = 2048  # Samples per frame
FRAMES_PER_FFT = 16

SAMPLES_PER_FFT = FRAME_SIZE * FRAMES_PER_FFT
FREQ_STEP = float(FSAMP) / SAMPLES_PER_FFT

NOTE_NAMES = 'E F F# G G# A A# B C C# D D#'.split()


def freq_to_number(f): return 64 + 12 * np.log2(f / 329.63)


def number_to_freq(n): return 329.63 * 2.0 ** ((n - 64) / 12.0)


def note_name(n):
    return NOTE_NAMES[n % NOTE_MIN % len(NOTE_NAMES)] + str(int(n / 12 - 1))


def note_to_fftbin(n): return number_to_freq(n) / FREQ_STEP


imin = max(0, int(np.floor(note_to_fftbin(NOTE_MIN - 1))))
imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX + 1))))

buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
num_frames = 0

# initialize audio
stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                channels=1,
                                rate=FSAMP,
                                input=True,
                                frames_per_buffer=FRAME_SIZE)
stream.start_stream()
# create window
window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, SAMPLES_PER_FFT, False)))
# intialize text
print('sampling at', FSAMP, 'Hz with max resolution of', FREQ_STEP, 'Hz')
print()
while stream.is_active():  # gathering data
    buf[:-FRAME_SIZE] = buf[FRAME_SIZE]  # shifting the buffer down and new data in
    buf[-FRAME_SIZE:] = np.fromstring(stream.read(FRAME_SIZE), np.int16)
    # run the FFT on the window buffer
    fft = np.fft.rfft(buf * window)
    # get frequency of maximum response in range
    freq = ((np.abs(fft[imin:imax]).argmax() + imin) * FREQ_STEP)  # argmax=pos of highest value
    n = freq_to_number(freq)  # note number and nearest note
    n0 = int(round(n))

    # console o/p once we have a full buffer
    num_frames += 1
    if num_frames >= FRAMES_PER_FFT:
        if freq > 170 or freq < 150:
            print('freq: {:7.2f} Hz      note: {:>3s}'.format(freq, note_name(n0)))
        else:
            n = freq_to_number(freq / 2)  # note number and nearest note
            n0 = int(round(n))
            print('freq : {:7.2f} Hz      note:  {:>3s}'.format(freq / 2, note_name(n0)))
