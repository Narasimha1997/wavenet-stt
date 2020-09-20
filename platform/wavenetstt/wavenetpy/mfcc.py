import os
import librosa
import numpy as np
import logging
from . import Meta

logging.basicConfig(level = logging.INFO)

class MFCC :

    @staticmethod
    def transpose_signals(signals) :
        return np.transpose(signals, [1, 0])
    
    @staticmethod
    def get_mfcc_representation(audio_path, mono, sample_rate = Meta.sample_rate) :

        if not os.path.exists(audio_path) :
            logging.error("File not found - {}".format(audio_path))
            return None
        waveform, _ = librosa.load(audio_path, mono = mono, sr = sample_rate)
        mfcc_signals = librosa.feature.mfcc(waveform, sr = sample_rate, n_mfcc = Meta.number_of_channels)
        return MFCC.transpose_signals(mfcc_signals)
    