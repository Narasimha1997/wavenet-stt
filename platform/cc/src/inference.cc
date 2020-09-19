#include <inference.h>


static inline std::string index_to_character(long * index_map, int& length) {
    std::string english = " abcdefghijklmnopqrstuvwxyz";
    std::string speechData;
    for(int i = 0; i < length; i++) {
        speechData.push_back(
            english.at(*(index_map + i))
        );
    }

    return speechData;
}


void InferenceCore::logError(tensorflow::Status * status) {
    if(status) {
        std::cout<<"Error["<<status->code() <<"] " << status->ToString() << std::endl;
    }
}

InferenceCore::InferenceCore(bool storeNames) {
    tensorflow::Status sessionStatus = tensorflow::NewSession(
        tensorflow::SessionOptions(), &this->wavenet_session
    );

    if(!sessionStatus.ok()) {
        std::cout<<"Failed to create tensorflow session"<<std::endl;
        this->logError(&sessionStatus);
        exit(0);
    }
    std::cout<<"Loaded tensorflow session"<<std::endl;
    this->graph_def = NULL;
    this->storeNames = storeNames;
}

bool InferenceCore::check_path(std::string * file_path) {
    struct stat buffer;
    return (stat(file_path->c_str(), &buffer) == 0);
}

void InferenceCore::readSavedModel(tensorflow::GraphDef * graph_def, std::string * file_path) {
    tensorflow::Status loadStats = tensorflow::ReadBinaryProto(
        tensorflow::Env::Default(), *file_path, graph_def
    );
    if (!loadStats.ok()) {
        std::cout<<"failed to load graph " << *file_path <<std::endl;
        this->logError(&loadStats);
        exit(0);
    }
}

void InferenceCore::LoadWavenet(std::string filePath) {
    if (!this->check_path(&filePath)) {
        std::cout<<"Graph " << filePath << "not found"<<std::endl;
        exit(0);
    }

    tensorflow::GraphDef graph_def;
    this->readSavedModel(&graph_def, &filePath);
    tensorflow::Status sessionStatus = this->wavenet_session->Create(
        graph_def
    );

    if (! sessionStatus.ok() ) {
        std::cout<<"Failed loading graph into session " <<std::endl;
        this->logError(&sessionStatus);
        exit(0);
    }

    std::cout<<"Loaded wavenet"<<std::endl;

    //store names in a vector if enabled
    if (this->storeNames) {
        int n_nodes = graph_def.node_size();
        for (int i = 0; i < n_nodes; i++) {
            auto node = graph_def.node(i);
            this->node_names.push_back(node.name());
        }
    }
}

std::vector<std::string> * InferenceCore::GetOpNames() {
    return &this->node_names;
}

std::vector<std::string> InferenceCore::Infer(const float * mfcc, int seq_length, int n_channels) {
    tensorflow::TensorShape mfcc_shape({1, seq_length, n_channels});
    tensorflow::TensorShape seq_length_shape({1});

    tensorflow::Tensor mfcc_tensor(tensorflow::DT_FLOAT, mfcc_shape);
    tensorflow::Tensor seq_length_tensor(tensorflow::DT_INT32, seq_length_shape);

    //create a output tensor:
    std::vector<tensorflow::Tensor> outputTensor;

    //TODO: Try to cast tensor without copy, more efficient way required
    memcpy(mfcc_tensor.data(), mfcc, seq_length * n_channels * sizeof(float));
    seq_length_tensor.vec<int>()(0) = seq_length;

    tensorflow::Status inferenceStatus = this->wavenet_session->Run(
       {{"mfcc:0", mfcc_tensor}, {"sequence_length:0", seq_length_tensor}},
       {{"output:0"}},
       {},
       &outputTensor
    );

    if (!inferenceStatus.ok()) {
        std::cout<<"Inference failed"<<std::endl;
        logError(&inferenceStatus);
        exit(0);
    }

    //postporcess beam search outputs to labels :
    std::vector<std::string> speechOutput;
    for(tensorflow::Tensor tensor : outputTensor) {
        //iterate the tensor and map outputs:
        tensorflow::TensorShape shape = tensor.shape();
        int n_rows = shape.dim_size(0);
        int sentenceLength = shape.dim_size(1);
        
        long * int_indexes = (long *)tensor.data();

        for(int i = 0; i < n_rows; i++) {
            speechOutput.push_back(
                index_to_character(int_indexes, sentenceLength)
            );
            int_indexes += sentenceLength;
        }
    }

    return speechOutput;
}

