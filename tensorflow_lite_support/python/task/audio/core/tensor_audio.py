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

from tensorflow_lite_support.python.task.audio.core.pybinds import audio_buffer


class TensorAudio(object):
  """A wrapper class to store the input audio."""

  def __init__(self,
               audio_format: audio_buffer.AudioFormat,
               sample_count: int,
               audio_data: audio_buffer.AudioBuffer = None,
               is_from_file: bool = False,
               ) -> None:
    """Initializes the `TensorAudio` object.

    Args:
      audio_format: audio_buffer.AudioFormat, format of the audio.
      sample_count: int, number of samples in the audio.
      audio_data: audio_buffer.AudioBuffer, contains raw audio data, buffer size
      and audio format info.
      is_from_file: boolean, whether `audio_data` is loaded from the audio file.
    """
    self._format = audio_format
    self._sample_count = sample_count
    self._is_from_file = is_from_file

    if self._is_from_file:
      self._data = audio_data
    else:
      self._buffer = np.zeros([self._sample_count, self._format.channels])

  def clear(self):
    """Clear the internal buffer and fill it with zeros."""
    self._buffer.fill(0)

  @classmethod
  def from_wav_file(cls,
                    file_name: str,
                    buffer_size: int) -> "TensorAudio":
    """Loads `audio_buffer.AudioBuffer` from the WAV file

    Args:
      file_name: WAV file name.
      buffer_size: Required input buffer size.
    Raises:
      status.StatusNotOk if the audio file can't be decoded. Need to import
        the module to catch this error: `from pybind11_abseil import status`,
        see https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    audio_data = audio_buffer.LoadAudioBufferFromFile(
      file_name, buffer_size, np.zeros([buffer_size]))
    decoded_buffer_size = audio_data.get_buffer_size()
    decoded_audio_format = audio_data.get_audio_format()

    return cls(decoded_audio_format, decoded_buffer_size, audio_data,
               is_from_file=True)

  def load_from_array(self, src: np.ndarray) -> None:
    """Load audio data from a NumPy array.
    Args:
      src: A NumPy array contains the input audio.
    """
    if len(src) > len(self._buffer):
      raise ValueError('Input audio is too large.')
    elif src.shape[1] != self._format.channels:
      raise ValueError('Input audio contains an invalid number of channels.')

    # Shift the internal buffer backward and add the incoming data to the end of
    # the buffer.
    shift = len(src)
    self._buffer = np.roll(self._buffer, -shift, axis=0)
    self._buffer[-shift:, :] = src

  def get_format(self) -> audio_buffer.AudioFormat:
    """Gets the audio format of the audio."""
    return self._format

  def get_sample_count(self) -> int:
    """Gets the sample count of the audio."""
    return self._sample_count

  def get_data(self) -> audio_buffer.AudioBuffer:
    """Gets the `audio_buffer.AudioBuffer` object."""
    if self._is_from_file:
      audio_data = self._data
    else:
      audio_data = audio_buffer.AudioBuffer(
        self._buffer, self._sample_count, self._format)
    return audio_data
