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
"""Tests for audio_record."""

import unittest
import numpy as np

from numpy import testing
from unittest import mock

from tensorflow_lite_support.python.task.audio.core import audio_record

_CHANNELS = 1
_SAMPLING_RATE = 16000
_BUFFER_SIZE = 15600


class AudioRecordTest(unittest.TestCase):
  def setUp(self):
    super().setUp()

    # Mock sounddevice.InputStream.
    self.mock_input_stream = mock.MagicMock()

    with mock.patch('sounddevice.InputStream',
                    return_value=self.mock_input_stream) as mock_input_stream:
      self.record = audio_record.AudioRecord(_CHANNELS, _SAMPLING_RATE,
                                             _BUFFER_SIZE)

      # Save the initialization arguments of InputStream for later assertion.
      self.assertEqual(mock_input_stream.call_count, 1)
      _, self.init_args = mock_input_stream.call_args

  def test_init_args(self):
    # Assert parameters of InputStream initialization.
    self.assertEqual(
      self.init_args['channels'], _CHANNELS,
      "InputStream's channels doesn't match the initialization argument.")
    self.assertEqual(
      self.init_args['samplerate'], _SAMPLING_RATE,
      "InputStream's samplerate doesn't match the initialization argument.")

  def test_life_cycle(self):
    # Assert start recording routine.
    self.record.start_recording()
    self.mock_input_stream.start.assert_called_once()
    self.assertTrue(self.mock_input_stream.active)

    # Assert stop recording routine.
    self.record.stop()
    self.mock_input_stream.stop.assert_called_once()
    self.assertTrue(self.mock_input_stream.stopped)

  def test_buffer_data(self):
    callback_fn = self.init_args['callback']

    # Create dummy data to feed to the AudioRecord instance.
    chunk_size = int(_BUFFER_SIZE * 0.5)
    input_data = []
    for _ in range(3):
      dummy_data = np.random.rand(chunk_size, 1).astype(float)
      input_data.append(dummy_data)
      callback_fn(dummy_data)

    # Assert read data of a single chunk.
    recorded_audio_data = self.record.read(chunk_size)
    testing.assert_almost_equal(recorded_audio_data, input_data[-1])

    # Assert read all data in buffer.
    recorded_audio_data = self.record.read(chunk_size * 2)
    expected_data = np.concatenate(input_data[-2:])
    testing.assert_almost_equal(recorded_audio_data, expected_data)

    # Assert exception if read too much data.
    with self.assertRaisesRegex(
        ValueError, 'Cannot read more samples than the size of the buffer.'):
      self.record.read(chunk_size * 3)


if __name__ == '__main__':
  unittest.main()
