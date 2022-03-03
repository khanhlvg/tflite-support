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

#include <iostream>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/audio/core/audio_buffer.h"
#include "tensorflow_lite_support/cc/task/audio/utils/wav_io.h"

namespace tflite {
namespace task {
namespace audio {

tflite::support::StatusOr<std::unique_ptr<AudioBuffer>>
DecodeAudioFromWaveFile(const std::string& wav_file) {
    std::string contents = ReadFile(wav_file);
    std::vector<float>* wav_data;
    uint32_t decoded_sample_count;
    uint16_t decoded_channel_count;
    uint32_t decoded_sample_rate;
    RETURN_IF_ERROR(DecodeLin16WaveAsFloatVector(
            contents, wav_data, &decoded_sample_count, &decoded_channel_count,
            &decoded_sample_rate));

    return AudioBuffer::Create(
            wav_data->data(), decoded_sample_count,
            {decoded_channel_count, static_cast<int>(decoded_sample_rate)});
}

}  // namespace audio
}  // namespace task
}  // namespace tflite
