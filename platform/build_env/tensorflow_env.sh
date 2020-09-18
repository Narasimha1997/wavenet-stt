#!/bin/bash

#change these variables
TENSORFLOW_CC_LIBS_PATH=${HOME}/Documents/installation/tensorflow/lib
TENSORFLOW_CC_INCLUDE_PATH=${HOME}/Documents/installation/tensorflow/include 
LIB_PYTHON_PATH=

#wavenet-sst-dir
present_dir=$(pwd)
SST_SOURCE_DIR=${present_dir}/../cc/src 
SST_INCLUDE_PATH=${present_dir}/../cc/include

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${TENSORFLOW_CC_LIBS_PATH}

if [[ $1 == "run" ]]; then 
    ./sst_test $2
    echo "Ran"
    exit
fi

if [[ $1 == "python" ]]; then
    PYTHON_EXT_SUFFIX=$(python3-config --extension-suffix)
    PYTHON_PYBIND_INCLUDES=$(python3 -m pybind11 --includes)
    g++ ${SST_SOURCE_DIR}/python/python_wrapper.cc ${SST_SOURCE_DIR}/inference.cc \
        -fPIC -shared \
        -I${SST_INCLUDE_PATH} -I${TENSORFLOW_CC_INCLUDE_PATH} \
        -I${TENSORFLOW_CC_INCLUDE_PATH}/src \
        -L${TENSORFLOW_CC_LIBS_PATH} \
        -ltensorflow_cc -lpthread -lprotobuf -lpython3.6m  \
        -Wall -Wextra \
        ${PYTHON_PYBIND_INCLUDES} \
        -o wavenetsst${PYTHON_EXT_SUFFIX}
        exit
fi 

#start compiling
g++ ../cc/main.cc ${SST_SOURCE_DIR}/inference.cc \
    -I${SST_INCLUDE_PATH} -I${TENSORFLOW_CC_INCLUDE_PATH} \
    -I${TENSORFLOW_CC_INCLUDE_PATH}/src \
    -L${TENSORFLOW_CC_LIBS_PATH} \
    -ltensorflow_cc -lpthread -lprotobuf  \
    -Wall -Wextra \
    -o sst_test 