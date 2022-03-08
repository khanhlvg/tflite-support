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
import unittest

from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.task.audio.core.pybinds import audio_buffer
from tensorflow_lite_support.python.test import test_util


class TensorAudioTest(parameterized.TestCase, unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.test_audio_path = test_util.get_test_data_path('speech.wav')

  def test_from_file(self):
    # Test data
    input_channels = 1
    input_sample_rate = 16000
    input_audio_format = tensor_audio.TensorAudioFormat(
      input_sample_rate, input_channels)
    input_sample_count = 15600

    # Load TensorAudio object.
    tensor = tensor_audio.TensorAudio.from_wav_file(
      self.test_audio_path, 15600)
    tensor_audio_format = tensor.get_format()

    self.assertEqual(tensor.get_sample_count(), input_sample_count)
    self.assertEqual(tensor_audio_format.channels, input_audio_format.channels)
    self.assertEqual(
      tensor_audio_format.sample_rate, input_audio_format.sample_rate)
    self.assertIsInstance(tensor.get_data(), audio_buffer.AudioBuffer)


if __name__ == '__main__':
  unittest.main()
