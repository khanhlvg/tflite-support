/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow_lite_support/examples/task/audio/desktop/utils/audio_utils.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil

namespace tflite {
namespace task {
namespace audio {

namespace {
namespace py = ::pybind11;

}  //  namespace


PYBIND11_MODULE(audio_utils, m) {
  // python wrapper for AudioData class which shouldn't be directly used by
  // the users.
  pybind11::google::ImportStatusModule();

  py::class_<AudioData>(m, "AudioData", py::buffer_protocol())
      .def(py::init([](py::buffer buffer, const int sample_rate) {
        py::buffer_info info = buffer.request();

        int sample_count = info.shape[0];
        int channels = info.shape[1];

        return AudioData{
            static_cast<float *>(info.ptr), sample_count, channels,
                         sample_rate};
      }))
      .def_readonly("sample_count", &AudioData::sample_count)
      .def_readonly("channels", &AudioData::channels)
      .def_readonly("sample_rate", &AudioData::sample_rate)
      .def_buffer([](AudioData &data) -> py::buffer_info {
        return py::buffer_info(
            data.wav_data, sizeof(float),
            py::format_descriptor<float>::format(), 2,
            {data.sample_count, data.channels},
            {sizeof(float) * size_t(data.channels),
             sizeof(float)});
      });

//  m.def("DecodeAudioFromWaveFile", &DecodeAudioFromWaveFile);
  m.def("DecodeAudioFromWaveFile",
        [](const std::string& wav_file, int buffer_size, py::buffer buffer) {
            py::buffer_info info = buffer.request();

            return DecodeAudioFromWaveFile(
                    wav_file, buffer_size,
                    static_cast<std::vector<float> *>(info.ptr));
        });
}

}  // namespace vision
}  // namespace task
}  // namespace tflite

