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
#include "tensorflow_lite_support/cc/task/audio/core/audio_buffer.h"

#include "pybind11/pybind11.h"
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/audio/utils/wav_io.h"

namespace tflite {
namespace task {
namespace audio {

namespace {
namespace py = ::pybind11;

}  //  namespace

tflite::support::StatusOr<AudioBuffer> LoadAudioBufferFromFile(
        const std::string& wav_file, int buffer_size,
        std::vector<float>* wav_data) {
    std::string contents = ReadFile(wav_file);

    uint32_t decoded_sample_count;
    uint16_t decoded_channel_count;
    uint32_t decoded_sample_rate;

    RETURN_IF_ERROR(DecodeLin16WaveAsFloatVector(
            contents, wav_data, &decoded_sample_count, &decoded_channel_count,
            &decoded_sample_rate));

    if (decoded_sample_count > buffer_size) {
        decoded_sample_count = buffer_size;
    }

    return AudioBuffer(
            wav_data->data(), static_cast<int>(decoded_sample_count),
            {decoded_channel_count, static_cast<int>(decoded_sample_rate)});
}

PYBIND11_MODULE(audio_buffer, m) {
    // python wrapper for AudioBuffer class which shouldn't be directly used by
    // the users.

    py::class_<AudioBuffer::AudioFormat>(m, "AudioFormat")
        .def_readonly("channels", &AudioBuffer::AudioFormat::channels)
        .def_readonly("sample_rate", &AudioBuffer::AudioFormat::sample_rate);

    py::class_<AudioBuffer>(m, "AudioBuffer")
        .def("get_audio_format", &AudioBuffer::GetAudioFormat)
        .def("get_buffer_size", &AudioBuffer::GetBufferSize)
        .def("get_float_buffer", &AudioBuffer::GetFloatBuffer);

    m.def("LoadAudioBufferFromFile",
        [](const std::string& wav_file, int buffer_size, py::buffer buffer)
        -> tflite::support::StatusOr<AudioBuffer> {
        py::buffer_info info = buffer.request();

        return LoadAudioBufferFromFile(
                wav_file, buffer_size,
                static_cast<std::vector<float> *>(info.ptr));
    });
}

}  // namespace vision
}  // namespace task
}  // namespace tflite

