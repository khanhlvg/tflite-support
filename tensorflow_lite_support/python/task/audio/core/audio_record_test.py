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

from numpy.testing import assert_almost_equal
from absl.testing import parameterized
from unittest import mock

from tensorflow_lite_support.python.task.audio.core import audio_record


class AudioRecordTest(parameterized.TestCase, unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.channels = 1
    self.sampling_rate = 16000
    self.buffer_size = 15600

  @mock.patch("tensorflow_lite_support.python.task.audio.core.audio_record.sd")
  def test_audio_record_read(self, *args):
    input_data = np.zeros([self.buffer_size, self.channels], dtype=float)
    record = audio_record.AudioRecord(
      self.channels, self.sampling_rate, self.buffer_size)

    # Replace input stream data with dummy data.
    record._stream.read.return_value = input_data.tobytes()

    # Start recording.
    record.start_recording()
    self.assertTrue(record._stream.active)

    # Reads audio data captured in the buffer.
    recorded_audio_data = record.read(self.buffer_size)

    # Stop recording.
    record.stop()
    self.assertTrue(record._stream.stopped)

    # Close the stream.
    record._stream.close()

    # Compare recorded audio data with the dummy array.
    assert_almost_equal(recorded_audio_data, input_data)


if __name__ == '__main__':
  unittest.main()
