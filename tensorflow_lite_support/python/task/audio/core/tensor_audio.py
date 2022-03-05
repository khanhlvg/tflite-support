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
from typing import NamedTuple

import numpy as np

from tensorflow_lite_support.python.task.audio.core.pybinds import audio_utils
from tensorflow_lite_support.python.task.audio.core.pybinds import audio_buffer


class AudioFormat(NamedTuple):
  """Format of the incoming audio."""
  channels: int
  sample_rate: int


class TensorAudio(object):
  """A wrapper class to store the input audio."""

  def __init__(self,
               audio_format: AudioFormat,
               sample_count: int,
               ) -> None:
    """Initializes the `TensorAudio` object."""
    self._sample_count = sample_count

    audio_format = audio_buffer.AudioFormat(
      audio_format.channels, audio_format.sample_rate)
    self._format = audio_format
    self.clear()

    # Gets the AudioBuffer object.

  def clear(self):
    """Clear the internal buffer and fill it with zeros."""
    self._buffer = np.zeros([self._sample_count, self._format.channels])

  def load_from_file(self,
                file_name: str,
                ) -> audio_buffer.AudioBuffer:
    """Loads `audio_buffer.AudioBuffer` from the WAV file

    Args:
      file_name: WAV file name.
    Returns:
      `audio_buffer.AudioBuffer` object.

    Raises:
      status.StatusNotOk if the audio file can't be decoded. Need to import
        the module to catch this error: `from pybind11_abseil import status`,
        see https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    # print("Buffer before decoding", self._buffer)
    audio_data = audio_utils.DecodeAudioFromWaveFile(
      file_name, self._sample_count, self._buffer)
    # print(audio_data.get_buffer_size(),
    #       audio_data.get_audio_format().channels,
    #       audio_data.get_audio_format().sample_rate,
    #       audio_data.get_float_buffer())
    # print("Buffer after decoding", self._buffer)
    # self._buffer = np.array(audio_data.get_float_buffer(), copy=False)
    # self.load_from_array(audio_data.get_float_buffer())
    return audio_data

  @property
  def format(self) -> audio_buffer.AudioFormat:
    return self._format

  @property
  def sample_count(self) -> int:
    return self._sample_count

  @property
  def buffer(self) -> np.ndarray:
    return self._buffer

  def get_data(self):
    """Gets the audio data."""
    return self._buffer, self.sample_count, self._format
