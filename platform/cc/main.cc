#include <inference.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout<<"Provide model path" << std::endl;
        exit(0);
    }

    std::string filePath(argv[1]);
    std::cout<<"Model path : " << filePath << std::endl;

    InferenceCore inf = InferenceCore(true);
    inf.LoadWavenet(filePath);

    std::vector<std::string> * node_names = inf.GetOpNames();
    for(std::string node : *node_names) std::cout<<node<<std::endl;
}