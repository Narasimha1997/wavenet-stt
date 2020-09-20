from wavenetpy import WavenetSTT

#load the model
wavenet = WavenetSTT('../../pb/wavenet-stt.pb')

#pass the audio file
result = wavenet.infer_on_file('test.wav')
print(result)