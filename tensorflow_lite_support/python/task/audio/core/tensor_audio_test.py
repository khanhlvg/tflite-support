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
import numpy as np
from numpy.testing import assert_almost_equal
from absl.testing import parameterized
import unittest

from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.task.audio.core.pybinds import _pywrap_audio_buffer
from tensorflow_lite_support.python.test import test_util

_CppAudioFormat = _pywrap_audio_buffer.AudioFormat
_CppAudioBuffer = _pywrap_audio_buffer.AudioBuffer


class TensorAudioTest(parameterized.TestCase, unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.test_audio_path = test_util.get_test_data_path('speech.wav')

  def test_from_file(self):
    # Test data
    input_channels = 1
    input_sample_rate = 16000
    input_audio_format = _CppAudioFormat(input_channels, input_sample_rate)
    input_sample_count = 15600

    # Loads TensorAudio object from WAV file.
    tensor = tensor_audio.TensorAudio.from_wav_file(
      self.test_audio_path, 15600)
    tensor_audio_format = tensor.get_format()

    self.assertEqual(tensor.get_sample_count(), input_sample_count)
    self.assertEqual(tensor_audio_format.channels, input_audio_format.channels)
    self.assertEqual(
      tensor_audio_format.sample_rate, input_audio_format.sample_rate)
    self.assertIsInstance(tensor.get_data(), _CppAudioBuffer)

  def test_load_from_array(self):
    # Test data
    input_channels = 1
    input_sample_rate = 16000
    input_audio_format = _CppAudioFormat(
      input_channels, input_sample_rate)
    input_sample_count = 15600

    array = np.random.random((input_sample_count, input_channels))

    # Loads TensorAudio object from a NumPy array.
    tensor = tensor_audio.TensorAudio(
      audio_format=input_audio_format, sample_count=input_sample_count)
    tensor.load_from_array(array)

    tensor_audio_data = tensor.get_data()
    tensor_audio_format = tensor_audio_data.audio_format

    self.assertEqual(tensor_audio_format.channels, input_channels)
    self.assertEqual(tensor_audio_format.sample_rate, input_sample_rate)
    self.assertEqual(tensor_audio_data.buffer_size, input_sample_count)
    self.assertIsInstance(tensor_audio_data, _CppAudioBuffer)
    assert_almost_equal(tensor_audio_data.float_buffer, array)

  def test_load_from_array_with_different_sample_count(self):
    # Load TensorAudio object from a NumPy array.
    input_sample_count = 10000
    tensor = tensor_audio.TensorAudio(
      audio_format=_CppAudioFormat(1, 16000), sample_count=15600)
    array = np.random.random((input_sample_count, 1))

    # Loads TensorAudio object from a NumPy array.
    tensor.load_from_array(array)
    tensor_audio_data = tensor.get_data()

    self.assertNotEqual(tensor_audio_data.buffer_size, input_sample_count)

  def test_load_from_array_fails_with_too_many_input_samples(self):
    # Fails loading TensorAudio object from a NumPy with a sample count
    # exceeding TensorAudio's internal buffer capacity.
    tensor = tensor_audio.TensorAudio(
      audio_format=_CppAudioFormat(1, 16000), sample_count=15000)
    array = np.random.random((20000, 1))

    with self.assertRaisesRegex(ValueError, r'Input audio is too large.'):
      tensor.load_from_array(array)


if __name__ == '__main__':
  unittest.main()
