import wavenetsst
from modules.mfcc import MFCC

model = wavenetsst.Wavenet("../pb/wavenet-stt.pb")

wave = MFCC.get_mfcc_representation('test/test.wav', mono = True)
wave = wave.reshape(wave.shape[0], wave.shape[1], order='F')
model.infer(wave, wave.shape[0], 20)