#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "inference.h"

namespace python = pybind11;

static inline const float* np_to_float_32(python::array_t<float>& np_array) {
    return reinterpret_cast<float*>(
        np_array.request().ptr
    );
}

class PyWrap__Wavenet {
    private:
        InferenceCore * inferCore;
    public:
        PyWrap__Wavenet(std::string& model_path) {
            this->inferCore = new InferenceCore(false);
            this->inferCore->LoadWavenet(model_path);
        }
        //TODO: Export this later
        std::vector<std::string> * PyWrap__GetOperations() {
            return this->inferCore->GetOpNames();
        }

        void PyWrap__Infer(python::array_t<float>& mfcc, int seq_length, int n_channels) {
            const float * mfcc_float = np_to_float_32(mfcc);

            //sample test code
            for(int i = 0; i < seq_length * n_channels; i++) {
                std::cout<<mfcc_float[i]<<" ";
            }
        }
};

PYBIND11_MODULE(wavenetsst, m) {
    python::class_<PyWrap__Wavenet>(m, "Wavenet")
        .def(python::init<std::string&>())
        .def("infer", &PyWrap__Wavenet::PyWrap__Infer);
}