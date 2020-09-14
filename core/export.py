import os 
import argparse
from modules.tf_graph_export import export_saved_graph

parser = argparse.ArgumentParser()

#add args
parser.add_argument('--ckpt', default = "./ckpt", type = str, required = False)
parser.add_argument('--output', default = '../pb', type = str, required = False)
parser.add_argument('--logs', default = '', type = str, required = False)
parser.add_argument('--switch_channel', default = "No", type = str, required = False)
parser.add_argument('--beam_search', default = "Yes", type = str, required = False)

if __name__ == "__main__":
    
    args = parser.parse_args()

    write_logs = True if args.logs != '' else False

    result = export_saved_graph(
        args.ckpt, 
        args.output, 
        write_tfboard_log = write_logs,
        log_dir = args.logs,
        include_beam_search = True if 'y' in args.beam_search.lower() else False,
        switch_channels_for_tflite = True if 'y' in args.switch_channel.lower() else False
    )

    if not result :
        print('Error while exporting, exiting')
    else :
        print('Done exporting, exiting')
    