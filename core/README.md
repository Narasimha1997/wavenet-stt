#### Wavenet Core module
This package deals with core wavenet speech-to-text functionalities like model export and AI level operations and has nothing do with model deployment or engineering.

You can edit this package to modify the graph operations or add your own op.
The definition of the wavenet model for speech-to-text is provided in `modules/wavenet.py`

The wavenet model requires Mel-spectograph of the audio waveform, we are using `librosa` for this reason. The library first samples the audio wave file to `16000 samples/sec` and then generates a MFCC representation. The mfcc is fed into the model and at the output, a `ctc_beam_search_decoder` is used. This tensorflow op uses beam search method to extract word indexes from the model output.

##### Exporting the saved model:
If you plan to use C++ Module or use recognition script provided by the package, then you have to export the saved model from the checkpoints provided. Run `exporter.py`, all default options will be used. You can check
```
python3 exporter.py --help
```
for various options provided. (Better stick with defaults if you are using C++ module as it is).
After this, you will see the exported model at `pb/wavenet-stt.pb`.

##### Running end-to-end speech recognition :
The exported model can be used for end-to-end speech recognition, for this we have provided a `recognition.py`, 
it is recommended not to use this in production, but you can use this script to test your exported model with changes.

To run execute the script below and provide `model_dir` and `wav_file` as arguments.
```
python3 recognize.py --model_dir=../pb --wav_file=test/test.py
```