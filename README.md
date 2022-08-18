# openCL delegate issue with models that output results from their intermediate nodes

This repo contains scripts and a tool to reproduce the openCL delegate issue with models that output results from their intermediate nodes. The openCL delegate generates all zero outputs in such cases.

Here is a very simple example. Consider the following tflite model:

![direct](https://user-images.githubusercontent.com/45400368/185452166-f8f5842b-b551-4c58-8da6-f63728f5a0bd.png)

If you infer this model using the openCL delegate, you will get all zeros for the `Identity` output (also, you will get wrong results from `Identity_1`).

Our experiments have revealed that the solution for this issue is to use an identity/neutral node (e.g. Relu) and build a fake branch:

![indirect](https://user-images.githubusercontent.com/45400368/185454466-0e88b9d0-188b-454a-8603-a221819eca50.png)

This fake branch helps us to get correct results from both `Identity` and `Identity_1` outputs.

## Building and converting the model
* `model_files` folder contains a very simple model (`direct_output.h5`) representing the above-mentioned issue and its corresponding tflite version (`fp32_direct_output.tflite`). 
  * You can also use `generate_dummy_model.py` to build the model and use `convert_model.py` to convert it to tflite.

## tflite_inference tool 
We have implemented a small tool to feed an input to our sample tflite model using `openCL` delegate.

### PREREQUISITES: ###
* Linux host computer
* Connectivity to the target device via adb
* Android NDK, version 22 or later
* CMake 3.18 or later

### BUILD INSTRUCTIONS ###
* Unzip the `tensorflow_lite_cpp_2_9_1_nightly.zip` file inside the `tflite_inference_tool` folder.
* In a terminal, from `tflite_inference_tool` folder:
```console
$ mkdir build
$ cd build
$ cmake -G "Unix Makefiles"
        -DCMAKE_SYSTEM_NAME=Android 
        -DANDROID_ABI=arm64-v8a 
        -DANDROID_STL=c++_shared 
        -DANDROID_NATIVE_API_LEVEL=27 
        -DCMAKE_VERBOSE_MAKEFILE=ON 
        -DCMAKE_TOOLCHAIN_FILE=<path-to-ndk>/build/cmake/android.toolchain.cmake 
        -DCMAKE_BUILD_TYPE=Release
        -DTensorFlowLite_ROOT=../tensorflow_lite_cpp_2_9_1_nightly ..
$ make
```
* Here, you must replace <path-to-ndk> with the absolute path of the ndk installed on your computer. If you installed NDK through Android studio, it is typically located at:
    `/home/<username>/Android/Sdk/ndk/<version>/` on Linux

* `tensorflow_lite_cpp_2_9_1_nightly` is TensorflowFlow Lite library (nightly version) package.
### Run INSTRUCTIONS ###
WARNING: This step will write to your `/data/local/tmp` folder on device. Please make sure existing files in that folder are backed up as needed.

In a terminal, from `tflite_inference_tool` folder:
```console
$ ./run_me.sh
```

The output should be something like this:
```console
INFO: Created TensorFlow Lite delegate for GPU.
INFO: Initialized TensorFlow Lite runtime.
VERBOSE: Replacing 3 node(s) with delegate (TfLiteGpuDelegateV2) node, yielding 1 partitions.
INFO: Initialized OpenCL-based API.
INFO: Created 1 GPU delegate kernels.
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
```

NOTE: If in `main.cpp` you change line 22 to `TfLiteModel* model = TfLiteModelCreateFromFile("./fp32_indirect_output.tflite");` (which means using the model with the fake branch) and build and run the project again, you will get non-zero (correct) results.
