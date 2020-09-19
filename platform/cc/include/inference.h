#ifndef __INFERENCE_H
#define __INFERENCE_H

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <sys/stat.h>


#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"


class InferenceCore {

    private:
    tensorflow::Session * wavenet_session;
    tensorflow::GraphDef * graph_def = NULL;
    bool storeNames = false;
    std::vector<std::string> node_names;
    bool check_path(std::string * path);
    void readSavedModel(tensorflow::GraphDef * graph, std::string * model_path);
    void logError(tensorflow::Status * status);

    public :
    InferenceCore(bool storeNames);
    void LoadWavenet(std::string modelPath);
    std::vector<std::string> * GetOpNames();
    std::vector<std::string> Infer(const float * mfcc, int seq_length, int n_channels);
};

#endif