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
"""TensorAudio class."""

import numpy as np

from tensorflow_lite_support.python.task.audio.core.pybinds import audio_utils
from tensorflow_lite_support.python.task.audio.core.pybinds import audio_buffer


class TensorAudio(object):
  """A wrapper class to store the input audio."""

  def __init__(self,
               audio_format: audio_buffer.AudioFormat,
               sample_count: int,
               audio_data: audio_utils.AudioData = None
               ) -> None:
    """Initializes the `TensorAudio` object."""
    self._format = audio_format
    self._sample_count = sample_count
    self._buffer = np.zeros([self._sample_count, self._format.channels])
    self._data = audio_data
    self.clear()

  def clear(self):
    """Clear the internal buffer and fill it with zeros."""
    self._buffer.fill(0)

  def load_from_file(self, file_name: str) -> audio_utils.AudioData:
    """Loads `audio_utils.AudioData` from the WAV file

    Args:
      file_name: WAV file name.
    Raises:
      status.StatusNotOk if the audio file can't be decoded. Need to import
        the module to catch this error: `from pybind11_abseil import status`,
        see https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    audio_data = audio_utils.DecodeAudioFromWaveFile(
      file_name, self._sample_count, self._buffer)
    self._data = audio_data

  def get_format(self) -> audio_buffer.AudioFormat:
    """Gets the audio format of the audio."""
    return self._format

  def get_sample_count(self) -> int:
    """Gets the sample count of the audio."""
    return self._sample_count

  def get_buffer(self) -> np.ndarray:
    """Gets the internal buffer of the audio."""
    return self._buffer

  def get_data(self) -> np.ndarray:
    """Gets the numpy array that represents `self._audio_data`.

    Returns:
      Numpy array that represents `self._audio_data` which is an
        `audio_utils.AudioData` object. To avoid copy, we will use
        `return np.array(..., copy = False)`. Therefore, this `TensorAudio`
        object should outlive the returned numpy array.
    """
    return np.array(self._data, copy=False)
