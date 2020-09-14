import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from .wavenet import bulid_wavenet
from . import Meta

import logging
logging.basicConfig(level = logging.INFO)

def __create_placeholders(switch_dims = False) :
    #create tensorflow placeholders for inputs
    if switch_dims :
        mfcc_input = tf.placeholder(tf.float32, [None, 1, Meta.number_of_channels], name = "mfcc")
        sequence_length = tf.placeholder(tf.int32, [None], name = "sequence_length")
        return (mfcc_input, sequence_length)
    else :
        mfcc_input = tf.placeholder(tf.float32, [1, None, Meta.number_of_channels], name = "mfcc")
        sequence_length = tf.placeholder(tf.int32, [None], name = "sequence_length")
        return (mfcc_input, sequence_length)

def __create_compute_graph(placeholders, include_beam_search = True) :

    mfcc, sequence_length = placeholders

    print(mfcc.shape)

    #data first output
    if mfcc.shape[0] == None :
        mfcc = tf.transpose(mfcc, perm = [1, 0, 2], name = "transpose")
    #define logits
    logits = bulid_wavenet(mfcc, len(Meta.vocabulary), is_training = False)

    #define beam search
    if include_beam_search :
        transpose = tf.transpose(logits, perm = [1, 0, 2])
        decoded, _ = tf.nn.ctc_beam_search_decoder(transpose, sequence_length, merge_repeated = False)
        output = tf.sparse.to_dense(decoded[0], name = "output")
        return output

    #define a identity operatio to return logit as a named tensor
    output = tf.identity(logits, name = "output")
    return output



def export_saved_graph(ckpt_dir, saved_model_dir, write_tfboard_log = True, log_dir = './logs', include_beam_search = True, switch_channels_for_tflite = False):

    if not os.path.exists(ckpt_dir) :
        logging.error("No checkpoint directory - {}".format(ckpt_dir))
        return False

    #load placeholders
    placeholders = __create_placeholders(switch_dims = switch_channels_for_tflite)
    op_graph = __create_compute_graph(placeholders, include_beam_search = include_beam_search)

    #run exporter witn tensorflow session:
    saver = tf.train.Saver()
    with tf.Session() as session :
        session.run(tf.global_variables_initializer())
        saver.restore(session, os.path.join(ckpt_dir, "buriburisuri"))

        if write_tfboard_log :
            summary = tf.summary.FileWriter(log_dir, session.graph)
        
        #write output graph
        frozen_model = tf.graph_util.convert_variables_to_constants(
            session,
            tf.get_default_graph().as_graph_def(),
            ["output"]
        )

        with tf.gfile.FastGFile(os.path.join(saved_model_dir, 'wavenet-stt.pb'), 'wb') as writer :
            writer.write(frozen_model.SerializeToString())
        return True



        

