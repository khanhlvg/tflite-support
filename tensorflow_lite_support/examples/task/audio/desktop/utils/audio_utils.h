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
#ifndef TENSORFLOW_LITE_SUPPORT_EXAMPLES_TASK_AUDIO_DESKTOP_AUDIO_CLASSIFIER_LIB_H_
#define TENSORFLOW_LITE_SUPPORT_EXAMPLES_TASK_AUDIO_DESKTOP_AUDIO_CLASSIFIER_LIB_H_

#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/audio/core/audio_buffer.h"

#include <vector>

namespace tflite {
namespace task {
namespace audio {

struct AudioData {
    float* wav_data;
    int sample_count;
    int channels;
    int sample_rate;
};

// Decodes audio from the WAV file.
tflite::support::StatusOr<AudioData>
DecodeAudioFromWaveFile(const std::string& wav_file, int buffer_size);

// Creates the AudioBuffer object from the AudioData object.
tflite::support::StatusOr<std::unique_ptr<AudioBuffer>>
CreateAudioBufferFromAudioData(const AudioData& audio);

}  // namespace audio
}  // namespace task
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_EXAMPLES_TASK_AUDIO_DESKTOP_AUDIO_CLASSIFIER_LIB_H_
