#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

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

        std::vector<std::string> PyWrap__Infer(python::array_t<float>& mfcc, int seq_length, int n_channels) {
            const float * mfcc_float = np_to_float_32(mfcc);
            std::vector<std::string> outputs = this->inferCore->Infer(mfcc_float, seq_length, n_channels);
            return outputs;
        }
};

PYBIND11_MODULE(wavenetsst, m) {
    python::class_<PyWrap__Wavenet>(m, "Wavenet")
        .def(python::init<std::string&>())
        .def("infer", &PyWrap__Wavenet::PyWrap__Infer);
}