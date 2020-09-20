import os
from .wavenetsst import Wavenet



class Meta :
    sample_rate = 16000
    number_of_channels = 20
    vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<EMP>']

from .mfcc import MFCC

class WavenetSTT :
    def __init__(self, saved_model_path) :
        if not os.path.exists(saved_model_path):
            print('saved model {} does not exist'.format(
                saved_model_path
            ))
            os._exit(0)
        self.wavenet = Wavenet(saved_model_path)
    
    def infer_on_file(self, audio_file) :
        if not os.path.exists(audio_file) or not audio_file.endswith('.wav') :
            print('Audio file not found or is not a wav file')
            os._exit(0)
        
        mfcc = MFCC.get_mfcc_representation(audio_file, mono = True)
        
        seq_length = mfcc.shape[0]
        n_channels = mfcc.shape[1]

        #flatten in row major order
        flattend_mfcc = mfcc.flatten(order = 'C')
        result = self.wavenet.infer(flattend_mfcc, seq_length, n_channels)
        return result


    
