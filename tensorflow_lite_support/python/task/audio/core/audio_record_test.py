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
import sounddevice as sd

from numpy.testing import assert_almost_equal
from unittest import mock

from tensorflow_lite_support.python.task.audio.core import audio_record


def query_devices(device=None):
  device_settings: dict = {
    "name": "Test audio device",
    "hostapi": 0,
    "max_input_channels": 2,
    "max_output_channels": 2,
    "default_low_input_latency": 0.00870751953125,
    "default_low_output_latency": 0.00870751953125,
    "default_high_input_latency": 0.034830078125,
    "default_high_output_latency": 0.034830078125,
    "default_samplerate": 44099.81494981215,
  }
  if device == None:
    return [device_settings]
  else:
    return device_settings


class AudioRecordTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.channels = 1
    self.sampling_rate = 16000
    self.buffer_size = 15600

  @mock.patch('tensorflow_lite_support.python.task.audio.core.audio_record.'
              'sd.query_devices', side_effect=query_devices)
  def test_audio_record_read(self, *args):
    # Ensure the test audio device is being used.
    self.assertEqual(sd.query_devices()[0]['name'], query_devices()[0]['name'])

    # Dummy audio input data.
    input_data = np.random.rand(15600, 1).astype(float)

    # Create a mock of sd.InputStream and pass in data to the callback.
    class MockInputStream(sd.InputStream):
      def __init__(self, callback=None, **kwargs):
        def audio_callback(data, *_):
          return callback(input_data, *_)
        super().__init__(callback=audio_callback, **kwargs)

    with mock.patch('tensorflow_lite_support.python.task.audio.core.'
                    'audio_record.sd.InputStream', side_effect=MockInputStream):
      record = audio_record.AudioRecord(
        self.channels, self.sampling_rate, self.buffer_size)

      # Start recording.
      record.start_recording()
      self.assertTrue(record._stream.active)

      # Stop recording.
      record.stop()
      self.assertTrue(record._stream.stopped)

      # Reads audio data captured in the buffer.
      recorded_audio_data = record.read(self.buffer_size)

      # Close the stream.
      record._stream.close()

      # Compare recorded audio data with the dummy array.
      assert_almost_equal(recorded_audio_data, input_data)


if __name__ == '__main__':
  unittest.main()
