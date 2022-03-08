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

#include "tensorflow_lite_support/cc/task/audio/audio_classifier.h"

#include "pybind11/pybind11.h"
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/audio/core/audio_buffer.h"

namespace tflite {
namespace task {
namespace audio {

namespace {
namespace py = ::pybind11;
}  // namespace

PYBIND11_MODULE(_pywrap_audio_classifier, m) {
  // python wrapper for C++ AudioClassifier class which shouldn't be directly used
  // by the users.
  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  py::class_<AudioClassifier>(m, "AudioClassifier")
      .def_static(
          "create_from_options",
          [](const AudioClassifierOptions& options) {
            return AudioClassifier::CreateFromOptions(options);
          })
      .def("classify",
           [](AudioClassifier& self, const AudioBuffer& audio)
                   -> tflite::support::StatusOr<ClassificationResult> {
               ASSIGN_OR_RETURN(
                       std::unique_ptr<AudioBuffer> audio_buffer,
                       AudioBuffer::Create(audio));
               return self.Classify(*audio_buffer);
           })
      .def("get_required_audio_format",
           &AudioClassifier::GetRequiredAudioFormat)
      .def("get_required_input_buffer_size",
           &AudioClassifier::GetRequiredInputBufferSize);
}

}  // namespace vision
}  // namespace task
}  // namespace tflite
