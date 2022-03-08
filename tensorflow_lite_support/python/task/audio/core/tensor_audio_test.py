# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tensor_audio."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.task.audio.core.pybinds import audio_utils
from tensorflow_lite_support.python.task.audio.core.pybinds import audio_buffer
from tensorflow_lite_support.python.test import test_util


class TensorAudioTest(tf.test.TestCase, parameterized.TestCase):

  def test_from_file(self):
    audio_file = test_util.get_test_data_path('speech.wav')

    # Test data
    audio_format = audio_buffer.AudioFormat(1, 16000)
    sample_count = 15600

    tensor = tensor_audio.TensorAudio(
      audio_format=audio_format, sample_count=15600)
    audio = tensor.load_from_file(audio_file)
    self.assertIsInstance(audio._audio_data, audio_utils.AudioData)
    self.assertEqual(audio.get_format(), audio_format)
    self.assertEqual(audio.get_sample_count(), sample_count)
    self.assertIsInstance(audio.get_data(), np.ndarray)


if __name__ == '__main__':
  tf.test.main()
