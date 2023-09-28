import numpy as np
import pyaudio


class Sound:
    sr = None
    stream = None
    pa = None

    def __init__(self, sr=48000):
        self.initialize(sr)

    def __del__(self):
        self.stream.close()

    def initialize(self, sr):
        self.sr = sr
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sr,
                frames_per_buffer=1024,
                output=True,
                )

    def make_sound(self, pitch, volume, duration):
        t = np.linspace(0., duration, int(self.sr * duration))
        x = self.volume * np.sin(2.0*np.pi*440.0*t)
        self.stream.write(x.astype(np.float32).tostring())
