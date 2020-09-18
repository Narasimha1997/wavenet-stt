#include <inference.h>


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

void InferenceCore::Infer(void * mfcc, int seq_length, int n_channels) {
    tensorflow::TensorShape mfcc_shape({1, seq_length, n_channels});
    tensorflow::TensorShape seq_length_shape({1, seq_length});

    //tensorflow::Tensor mfcc(tensorflow::DT_FLOAT, mfcc_shape);
    //tensorflow::Tensor seq_length(tensorflow::DT_FLOAT, seq_length_shape);
}

