import wavenetsst
from modules.mfcc import MFCC

model = wavenetsst.Wavenet("../pb/wavenet-stt.pb")

wave = MFCC.get_mfcc_representation('test/Welcome.wav', mono = True)
shape = wave.shape[0]
wave = wave.flatten()

output = model.infer(wave, shape, 20)
print(output)