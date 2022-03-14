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

#include "tensorflow_lite_support/cc/task/audio/audio_embedder.h"

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

PYBIND11_MODULE(_pywrap_audio_embedder, m) {
  // python wrapper for C++ AudioEmbedder class which shouldn't be directly used
  // by the users.
  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  py::class_<AudioEmbedder>(m, "AudioEmbedder")
      .def_static(
          "create_from_options",
          [](const AudioEmbedderOptions& options) {
            return AudioEmbedder::CreateFromOptions(options);
          })
      .def("embed",
           [](AudioEmbedder& self, const AudioBuffer& audio)
                   -> tflite::support::StatusOr<
                           tflite::task::processor::EmbeddingResult> {
               return self.Embed(audio);
           })
      .def("get_required_audio_format",
           &AudioEmbedder::GetRequiredAudioFormat)
      .def("get_required_input_buffer_size",
           &AudioEmbedder::GetRequiredInputBufferSize);
}

}  // namespace vision
}  // namespace task
}  // namespace tflite
