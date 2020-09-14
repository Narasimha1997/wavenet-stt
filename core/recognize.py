import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
import argparse
from modules.mfcc import MFCC
from modules import Meta

def __read_pb(model_dir) :

    g_file = os.path.join(model_dir, 'wavenet-stt.pb')
    if not os.path.exists(g_file) :
        print('Export the model first, or provide a correct directory')
        os._exit(0)
    
    with tf.gfile.FastGFile(g_file, 'rb') as greader :
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(greader.read())
        return graph_def

def __get_inputs(graph) :
    return (
        graph.get_tensor_by_name("mfcc:0"),
        graph.get_tensor_by_name("sequence_length:0")
    )

def __get_output(graph) :
    return graph.get_tensor_by_name("output:0")

def __read_wave(wave_file) :

    wave = MFCC.get_mfcc_representation(wave_file, mono = True)
    return wave, wave.shape[0]


def __map_outputs_to_language(outputs) :
    sentences = []

    def map_index_to_char(index) :
        return Meta.vocabulary[index]

    for output in outputs :
        sentence = list(map(map_index_to_char, output))
        sentences.append("".join(sentence))
    
    return sentences

def infer(model_dir, wave_file) :
    if not os.path.exists(wave_file) :
        print('Wavefile not found')
        os._exit(0)

    with tf.Session() as sess :
        graph_def = __read_pb(model_dir)
        sess.graph.as_default()

        tf.import_graph_def(graph_def, name='')
        mfcc, seq_length = __get_inputs(sess.graph)
        output = __get_output(sess.graph)

        wave, seq = __read_wave(wave_file)

        outputs = sess.run(output, feed_dict = {mfcc : [wave], seq_length : [seq]})

        print(outputs)

        sentences = __map_outputs_to_language(outputs)
        print(sentences)
        print('Done')


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default = '../pb', type = str, required = False)
parser.add_argument('--wav_file', default = './test/test.wav', type = str, required = False)

if __name__ == "__main__":

    args = parser.parse_args()
    infer(args.model_dir, args.wav_file)
